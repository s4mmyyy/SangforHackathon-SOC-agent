"""Conservative, schema-agnostic ClickHouse catalog discovery and semantic mapping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from re import fullmatch
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


_IDENTIFIER_RE = r"[A-Za-z_][A-Za-z0-9_]*"
_METADATA_DATABASES_SQL = "SELECT name FROM system.databases"
_METADATA_TABLES_SQL = "SELECT database, name, engine FROM system.tables WHERE is_temporary = 0"
_METADATA_COLUMNS_SQL = "SELECT database, table, name, type FROM system.columns"


@runtime_checkable
class QueryClientProtocol(Protocol):
    """Minimal injectable query-client interface; no ClickHouse package is required."""

    def query(self, sql: str, **kwargs: Any) -> Any:
        """Execute a read-only query and return client-specific result data."""


def is_valid_identifier(identifier: str) -> bool:
    return bool(fullmatch(_IDENTIFIER_RE, identifier))


def quote_identifier(identifier: str) -> str:
    """Quote one validated identifier, never a dotted path supplied by a caller."""
    if not is_valid_identifier(identifier):
        raise ValueError(f"invalid SQL identifier: {identifier!r}")
    return f"`{identifier}`"


def quote_qualified_identifier(database: str, table: str) -> str:
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def normalize_query_rows(result: Any) -> list[dict[str, Any]]:
    """Normalize simple fake/client result shapes to list-of-dictionaries."""
    if result is None:
        return []
    if isinstance(result, Mapping):
        return [dict(result)]
    if isinstance(result, (list, tuple)):
        return _normalize_rows(result)

    for attribute in ("result_rows", "named_results"):
        if hasattr(result, attribute):
            value = getattr(result, attribute)
            if callable(value):
                value = value()
            if isinstance(value, Mapping):
                return [dict(value)]
            if isinstance(value, (list, tuple)):
                return _normalize_rows(value, getattr(result, "column_names", None))
    raise TypeError("query result must be a mapping, a row sequence, or expose result_rows/named_results")


def _normalize_rows(rows: Iterable[Any], column_names: Any = None) -> list[dict[str, Any]]:
    names = list(column_names) if isinstance(column_names, (list, tuple)) else None
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
        elif hasattr(row, "_asdict"):
            normalized.append(dict(row._asdict()))
        elif isinstance(row, (list, tuple)) and names and len(row) == len(names):
            normalized.append(dict(zip(names, row, strict=True)))
        else:
            raise TypeError("rows must be mappings or named rows; positional rows need column_names")
    return normalized


class ColumnInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database: str
    table: str
    name: str
    type: str

    @field_validator("database", "table", "name")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not is_valid_identifier(value):
            raise ValueError(f"invalid discovered identifier: {value!r}")
        return value


class TableInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database: str
    name: str
    engine: str | None = None
    columns: list[ColumnInfo] = Field(default_factory=list)

    @field_validator("database", "name")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not is_valid_identifier(value):
            raise ValueError(f"invalid discovered identifier: {value!r}")
        return value


class CatalogInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    databases: list[str] = Field(default_factory=list)
    tables: list[TableInfo] = Field(default_factory=list)

    @field_validator("databases")
    @classmethod
    def validate_databases(cls, values: list[str]) -> list[str]:
        for value in values:
            if not is_valid_identifier(value):
                raise ValueError(f"invalid discovered identifier: {value!r}")
        return sorted(set(values))


class MappingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic: str
    column: ColumnInfo
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class TableMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database: str
    table: str
    fields: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity_margin: float = Field(ge=0.0, le=1.0)
    candidates: dict[str, list[MappingCandidate]] = Field(default_factory=dict)

    @field_validator("database", "table")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not is_valid_identifier(value):
            raise ValueError(f"invalid discovered identifier: {value!r}")
        return value

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, fields: dict[str, str]) -> dict[str, str]:
        for value in fields.values():
            if not is_valid_identifier(value):
                raise ValueError(f"invalid discovered identifier: {value!r}")
        return fields


@dataclass(frozen=True)
class DiscoveryThresholds:
    minimum_score: float = 0.70
    ambiguity_margin: float = 0.15


_SEMANTIC_TOKENS: dict[str, frozenset[str]] = {
    "timestamp": frozenset({"timestamp", "time", "datetime", "date", "eventtime", "utctime", "created", "observed"}),
    "host": frozenset({"host", "hostname", "computer", "machine", "device", "endpoint"}),
    "ip": frozenset({"ip", "address", "remoteaddr", "localaddr", "src", "dst", "source", "destination"}),
    "process_guid": frozenset({"processguid", "process_guid", "processuuid", "process_uuid", "processidguid"}),
    "process_id": frozenset({"processid", "process_id", "pid", "parentprocessid", "parent_pid"}),
    "image": frozenset({"image", "executable", "exe", "processpath", "process_path", "binary"}),
    "command_line": frozenset({"commandline", "command_line", "cmdline", "cmd", "arguments", "args"}),
    "event_id": frozenset({"eventid", "event_id", "eventcode", "event_code", "eventtype", "event_type"}),
    "file": frozenset({"file", "filename", "file_name", "filepath", "file_path", "targetfilename", "target_file"}),
    "user": frozenset({"user", "username", "user_name", "account", "accountname", "account_name", "principal"}),
    "domain": frozenset({"domain", "dns", "hostname", "queryname", "query_name", "fqdn"}),
    "security_action": frozenset({"action", "verdict", "disposition", "blocked", "decision", "controlaction"}),
}

_STRING_TYPES = ("string", "fixedstring", "uuid", "ipv4", "ipv6", "lowcardinality")
_NUMERIC_TYPES = ("int", "uint", "float", "decimal")
_TIME_TYPES = ("date", "datetime", "timestamp")


def _tokens(value: str) -> set[str]:
    compact = "".join(character.lower() if character.isalnum() else "_" for character in value)
    parts = {part for part in compact.split("_") if part}
    parts.add(compact.replace("_", ""))
    return parts


def score_column(semantic: str, column: ColumnInfo) -> MappingCandidate:
    """Score only generic lexical/type signals, avoiding customer-schema assumptions."""
    terms = _SEMANTIC_TOKENS[semantic]
    name_tokens = _tokens(column.name)
    lowered_name = column.name.lower().replace("_", "")
    lowered_type = column.type.lower()
    reasons: list[str] = []
    score = 0.0

    exact = any(token.replace("_", "") == lowered_name for token in terms)
    if exact:
        score += 0.70
        reasons.append("exact generic semantic name match")
    elif name_tokens.intersection(terms):
        score += 0.45
        reasons.append("generic semantic token match")
    elif any(term.replace("_", "") in lowered_name for term in terms if len(term) > 3):
        score += 0.30
        reasons.append("generic semantic name fragment match")

    if semantic == "timestamp" and any(token in lowered_type for token in _TIME_TYPES):
        score += 0.30
        reasons.append("time-compatible type")
    elif semantic == "process_id" or semantic == "event_id":
        if any(token in lowered_type for token in _NUMERIC_TYPES):
            score += 0.20
            reasons.append("numeric-compatible type")
    elif semantic == "ip" and ("ipv4" in lowered_type or "ipv6" in lowered_type):
        score += 0.35
        reasons.append("IP-compatible type")
    elif semantic != "timestamp" and any(token in lowered_type for token in _STRING_TYPES):
        score += 0.10
        reasons.append("text-compatible type")

    return MappingCandidate(
        semantic=semantic,
        column=column,
        score=min(score, 1.0),
        reasons=reasons,
    )


class SchemaDiscovery:
    """Discover metadata and select only unambiguous semantic field mappings."""

    def __init__(
        self,
        client: QueryClientProtocol,
        thresholds: DiscoveryThresholds = DiscoveryThresholds(),
    ) -> None:
        self.client = client
        self.thresholds = thresholds

    def discover_catalog(self) -> CatalogInfo:
        databases = [row["name"] for row in normalize_query_rows(self.client.query(_METADATA_DATABASES_SQL)) if isinstance(row.get("name"), str) and is_valid_identifier(row["name"])]
        table_rows = normalize_query_rows(self.client.query(_METADATA_TABLES_SQL))
        column_rows = normalize_query_rows(self.client.query(_METADATA_COLUMNS_SQL))
        columns_by_table: dict[tuple[str, str], list[ColumnInfo]] = {}
        for row in column_rows:
            try:
                column = ColumnInfo.model_validate(row)
            except (TypeError, ValueError):
                continue
            columns_by_table.setdefault((column.database, column.table), []).append(column)

        tables: list[TableInfo] = []
        for row in table_rows:
            database, name = row.get("database"), row.get("name")
            if not isinstance(database, str) or not isinstance(name, str):
                continue
            try:
                tables.append(
                    TableInfo(
                        database=database,
                        name=name,
                        engine=row.get("engine") if isinstance(row.get("engine"), str) else None,
                        columns=columns_by_table.get((database, name), []),
                    )
                )
            except ValueError:
                continue
        return CatalogInfo(databases=databases, tables=tables)

    def mapping_candidates(self, table: TableInfo, semantic: str) -> list[MappingCandidate]:
        if semantic not in _SEMANTIC_TOKENS:
            raise ValueError(f"unsupported semantic: {semantic}")
        return sorted(
            (score_column(semantic, column) for column in table.columns),
            key=lambda candidate: candidate.score,
            reverse=True,
        )

    def select_mapping(self, table: TableInfo) -> TableMapping:
        fields: dict[str, str] = {}
        candidate_map: dict[str, list[MappingCandidate]] = {}
        accepted_scores: list[float] = []
        accepted_margins: list[float] = []
        for semantic in _SEMANTIC_TOKENS:
            candidates = self.mapping_candidates(table, semantic)
            candidate_map[semantic] = candidates[:3]
            if not candidates:
                continue
            best = candidates[0]
            runner_up = candidates[1].score if len(candidates) > 1 else 0.0
            margin = best.score - runner_up
            if best.score >= self.thresholds.minimum_score and margin >= self.thresholds.ambiguity_margin:
                fields[semantic] = best.column.name
                accepted_scores.append(best.score)
                accepted_margins.append(margin)
        return TableMapping(
            database=table.database,
            table=table.name,
            fields=fields,
            confidence=round(sum(accepted_scores) / len(accepted_scores), 3) if accepted_scores else 0.0,
            ambiguity_margin=round(min(accepted_margins), 3) if accepted_margins else 0.0,
            candidates=candidate_map,
        )

    def discover_mappings(self, catalog: CatalogInfo | None = None) -> list[TableMapping]:
        catalog = catalog or self.discover_catalog()
        return [self.select_mapping(table) for table in catalog.tables]
