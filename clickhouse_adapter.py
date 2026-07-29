"""Restricted, injected-client ClickHouse investigation adapter; it never connects itself."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evidence_models import (
    AttackStage,
    AuditTrace,
    EntityKind,
    EntityRef,
    EvidenceOutcome,
    EvidenceRecord,
    EvidenceSource,
    InvestigationTask,
    QueryResult,
    QueryStatus,
)
from schema_discovery import (
    QueryClientProtocol,
    TableMapping,
    is_valid_identifier,
    normalize_query_rows,
    quote_identifier,
    quote_qualified_identifier,
)


OBJECTIVE_REQUIREMENTS: dict[str, frozenset[str]] = {
    "process_investigation": frozenset({"timestamp"}),
    "file_investigation": frozenset({"timestamp"}),
    "network_investigation": frozenset({"timestamp"}),
    "dns_investigation": frozenset({"timestamp"}),
    "security_control_investigation": frozenset({"timestamp"}),
}

_ENTITY_SEMANTICS: dict[EntityKind, tuple[str, ...]] = {
    EntityKind.HOST: ("host",),
    EntityKind.IP: ("ip",),
    EntityKind.PROCESS_GUID: ("process_guid",),
    EntityKind.PROCESS_ID: ("process_id",),
    EntityKind.IMAGE: ("image",),
    EntityKind.COMMAND_LINE: ("command_line",),
    EntityKind.EVENT_ID: ("event_id",),
    EntityKind.FILE: ("file",),
    EntityKind.USER: ("user",),
    EntityKind.DOMAIN: ("domain",),
}


class AdapterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rows: int = Field(default=1_000, ge=1, le=100_000)
    max_window_seconds: int = Field(default=86_400, ge=1, le=31 * 86_400)
    query_settings: dict[str, Any] = Field(
        default_factory=lambda: {"readonly": 1, "max_execution_time": 30}
    )
    minimum_mapping_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    minimum_mapping_margin: float = Field(default=0.15, ge=0.0, le=1.0)


class BuiltQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    mapping: TableMapping
    selected_fields: list[str] = Field(default_factory=list)
    audit: list[AuditTrace] = Field(default_factory=list)


class ClickHouseAdapter:
    """Build and execute only bounded, mapping-validated SELECT investigations."""

    def __init__(
        self,
        client: QueryClientProtocol,
        mappings: list[TableMapping],
        settings: AdapterSettings | None = None,
    ) -> None:
        self.client = client
        self.mappings = list(mappings)
        self.settings = settings or AdapterSettings()

    def execute(self, task: InvestigationTask) -> QueryResult:
        query_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        try:
            built = self.build_query(task, query_id=query_id)
        except ValueError as exc:
            return self._declined(query_id, started_at, str(exc), task)

        try:
            result = self._call_client(built.sql, built.parameters)
            rows = normalize_query_rows(result)
        except Exception as exc:  # Client APIs intentionally vary; report safely to caller.
            audit = built.audit + [
                AuditTrace(
                    event="query_execution_error",
                    success=False,
                    query_id=query_id,
                    details={"error_type": type(exc).__name__},
                )
            ]
            return QueryResult(
                query_id=query_id,
                status=QueryStatus.ERROR,
                reason=f"query execution failed: {type(exc).__name__}",
                rows=[],
                row_count=0,
                sql=built.sql,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                audit=audit,
            )

        bounded_rows = rows[: min(task.max_rows, self.settings.max_rows)]
        evidence = self._rows_to_evidence(bounded_rows, built.mapping, query_id)
        audit = built.audit + [
            AuditTrace(
                event="query_executed",
                success=True,
                query_id=query_id,
                details={"returned_rows": len(bounded_rows)},
            )
        ]
        return QueryResult(
            query_id=query_id,
            status=QueryStatus.SUCCESS,
            rows=bounded_rows,
            evidence=evidence,
            row_count=len(bounded_rows),
            sql=built.sql,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            audit=audit,
        )

    def build_query(self, task: InvestigationTask, *, query_id: str | None = None) -> BuiltQuery:
        if task.objective not in OBJECTIVE_REQUIREMENTS:
            raise ValueError("unsupported investigation objective")
        if task.max_rows > self.settings.max_rows:
            raise ValueError("task max_rows exceeds adapter bounded limit")
        if (task.time_end - task.time_start).total_seconds() > self.settings.max_window_seconds:
            raise ValueError("task time window exceeds adapter bounded limit")
        mapping = self._select_mapping(task)
        required = OBJECTIVE_REQUIREMENTS[task.objective]
        if not required.issubset(mapping.fields):
            raise ValueError("selected mapping lacks required timestamp field")

        timestamp = mapping.fields["timestamp"]
        validated_fields = self._validated_mapping_fields(mapping)
        selected = [field for field in validated_fields.values()]
        where = [
            f"{quote_identifier(timestamp)} >= {{time_start:DateTime64(3, 'UTC')}}",
            f"{quote_identifier(timestamp)} < {{time_end:DateTime64(3, 'UTC')}}",
        ]
        parameters: dict[str, Any] = {
            "time_start": task.time_start.astimezone(timezone.utc),
            "time_end": task.time_end.astimezone(timezone.utc),
        }
        entity_filters = self._entity_filters(task.target_entities, mapping, parameters)
        if task.target_entities and not entity_filters:
            raise ValueError("no compatible discovered entity filter is available")
        where.extend(entity_filters)

        limit = min(task.max_rows, self.settings.max_rows)
        sql = (
            "SELECT "
            + ", ".join(quote_identifier(field) for field in selected)
            + " FROM "
            + quote_qualified_identifier(mapping.database, mapping.table)
            + " WHERE "
            + " AND ".join(where)
            + f" LIMIT {limit}"
        )
        query_id = query_id or str(uuid4())
        return BuiltQuery(
            sql=sql,
            parameters=parameters,
            mapping=mapping,
            selected_fields=selected,
            audit=[
                AuditTrace(
                    event="query_built",
                    success=True,
                    query_id=query_id,
                    details={
                        "objective": task.objective,
                        "table": f"{mapping.database}.{mapping.table}",
                        "limit": limit,
                        "entity_filter_count": len(entity_filters),
                        "settings": self.settings.query_settings,
                    },
                )
            ],
        )

    def _select_mapping(self, task: InvestigationTask) -> TableMapping:
        eligible = [
            mapping
            for mapping in self.mappings
            if mapping.confidence >= self.settings.minimum_mapping_confidence
            and mapping.ambiguity_margin >= self.settings.minimum_mapping_margin
            and "timestamp" in mapping.fields
            and OBJECTIVE_REQUIREMENTS[task.objective].issubset(mapping.fields)
            and self._mapping_identifiers_valid(mapping)
        ]
        if not eligible:
            raise ValueError("no discovered mapping satisfies confidence, ambiguity, table, and timestamp requirements")
        if len(eligible) != 1:
            raise ValueError("mapping selection is ambiguous; a single suitable discovered mapping is required")
        return eligible[0]

    @staticmethod
    def _mapping_identifiers_valid(mapping: TableMapping) -> bool:
        return (
            is_valid_identifier(mapping.database)
            and is_valid_identifier(mapping.table)
            and all(is_valid_identifier(name) for name in mapping.fields.values())
        )

    @staticmethod
    def _validated_mapping_fields(mapping: TableMapping) -> dict[str, str]:
        fields = {semantic: name for semantic, name in mapping.fields.items() if is_valid_identifier(name)}
        if not fields or "timestamp" not in fields:
            raise ValueError("mapping contains no valid timestamp field")
        return fields

    def _entity_filters(
        self,
        entities: list[EntityRef],
        mapping: TableMapping,
        parameters: dict[str, Any],
    ) -> list[str]:
        filters: list[str] = []
        for index, entity in enumerate(entities):
            semantics = _ENTITY_SEMANTICS.get(entity.kind, ())
            matching_field = next((mapping.fields[name] for name in semantics if name in mapping.fields), None)
            if matching_field is None:
                continue
            parameter_name = f"entity_{index}"
            # Values are passed separately; entity values never alter SQL structure.
            parameters[parameter_name] = entity.value
            filters.append(f"{quote_identifier(matching_field)} = {{{parameter_name}:String}}")
        return filters

    def _call_client(self, sql: str, parameters: dict[str, Any]) -> Any:
        try:
            return self.client.query(sql, parameters=parameters, settings=self.settings.query_settings)
        except TypeError:
            try:
                return self.client.query(sql, parameters=parameters)
            except TypeError:
                # Simple fake clients may not accept parameters. Render only known placeholders
                # with escaped literals; identifiers remain independently validated and quoted.
                return self.client.query(self._literalized_sql(sql, parameters))

    @staticmethod
    def _literalized_sql(sql: str, parameters: dict[str, Any]) -> str:
        rendered = sql
        for name, value in parameters.items():
            if isinstance(value, datetime):
                utc_value = value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
                literal = f"toDateTime64('{utc_value}', 3, 'UTC')"
                placeholder = f"{{{name}:DateTime64(3, 'UTC')}}"
            else:
                escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
                literal = f"'{escaped}'"
                placeholder = f"{{{name}:String}}"
            if placeholder not in rendered:
                raise ValueError(f"missing expected query placeholder: {name}")
            rendered = rendered.replace(placeholder, literal)
        if "{" in rendered or "}" in rendered:
            raise ValueError("unrendered query placeholders are not allowed")
        return rendered

    def _rows_to_evidence(
        self, rows: list[dict[str, Any]], mapping: TableMapping, query_id: str
    ) -> list[EvidenceRecord]:
        reverse_fields = {column: semantic for semantic, column in mapping.fields.items()}
        evidence: list[EvidenceRecord] = []
        for index, row in enumerate(rows):
            entities: list[EntityRef] = []
            for column, value in row.items():
                semantic = reverse_fields.get(column)
                kind = self._entity_kind(semantic)
                if kind is not None and value is not None:
                    entities.append(EntityRef(kind=kind, value=str(value)))
            observed_at = self._row_timestamp(row.get(mapping.fields.get("timestamp", "")))
            action_value = row.get(mapping.fields.get("security_action", ""))
            evidence.append(
                EvidenceRecord(
                    evidence_id=f"{query_id}:{index}",
                    source=EvidenceSource.QUERY,
                    observed_at=observed_at,
                    entities=entities,
                    attack_stage=self._stage_for_objective(mapping),
                    outcome=self._explicit_action_outcome(action_value),
                    confidence=mapping.confidence,
                    attributes={
                        "mapping_table": f"{mapping.database}.{mapping.table}",
                        "security_control_blocked": self._is_explicit_block_action(action_value),
                        "query_row": row,
                    },
                    raw_reference=f"query:{query_id}:row:{index}",
                    query_id=query_id,
                )
            )
        return evidence

    @staticmethod
    def _is_explicit_block_action(value: Any) -> bool:
        """Accept only unambiguous security-control verdicts from a mapped action field."""
        if not isinstance(value, str):
            return False
        return value.strip().lower() in {"block", "blocked", "deny", "denied", "intercepted"}

    @classmethod
    def _explicit_action_outcome(cls, value: Any) -> EvidenceOutcome:
        """Do not infer outcomes from event IDs, HTTP fields, table names, or free-form rows."""
        return EvidenceOutcome.BLOCKED if cls._is_explicit_block_action(value) else EvidenceOutcome.UNKNOWN

    @staticmethod
    def _row_timestamp(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        return None

    @staticmethod
    def _entity_kind(semantic: str | None) -> EntityKind | None:
        try:
            return EntityKind(semantic) if semantic in EntityKind._value2member_map_ else None
        except TypeError:
            return None

    @staticmethod
    def _stage_for_objective(mapping: TableMapping) -> AttackStage:
        # Mapping itself carries no outcome; use unknown rather than derive event semantics from table names.
        del mapping
        return AttackStage.UNKNOWN

    @staticmethod
    def _declined(
        query_id: str, started_at: datetime, reason: str, task: InvestigationTask
    ) -> QueryResult:
        audit = AuditTrace(
            event="query_declined",
            success=False,
            query_id=query_id,
            details={"objective": task.objective, "reason": reason},
        )
        return QueryResult(
            query_id=query_id,
            status=QueryStatus.DECLINED,
            reason=reason,
            rows=[],
            row_count=0,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            audit=[audit],
        )
