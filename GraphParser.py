"""Defensive NDR event-graph parsing and evidence-contract integration."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal

from alert_intent_parser import AlertEntity, AlertSemantics, EntityType, StructuredAlert
from evidence_models import (
    AttackStage,
    CaseContext,
    EntityKind,
    EntityRef,
    EvidenceOutcome,
    EvidenceRecord,
    EvidenceSource,
)


class NDRGraphParser:
    """Normalize NDR graph JSON without invoking or depending on an LLM."""

    _VERTEX_KINDS = {
        "ip": EntityKind.IP,
        "domain": EntityKind.DOMAIN,
        "dns": EntityKind.DOMAIN,
        "file": EntityKind.FILE,
        "process": EntityKind.IMAGE,
        "host": EntityKind.HOST,
        "hostname": EntityKind.HOST,
        "user": EntityKind.USER,
        "url": EntityKind.URL,
        "alert": EntityKind.ALERT,
    }
    _ALERT_ENTITY_TYPES = {
        EntityKind.IP: "IP",
        EntityKind.DOMAIN: "DOMAIN",
        EntityKind.FILE: "FILE",
        EntityKind.IMAGE: "PROCESS",
        EntityKind.USER: "USER",
        EntityKind.URL: "URL",
    }
    _STAGE_VALUES = {
        "recon": AttackStage.RECON,
        "reconnaissance": AttackStage.RECON,
        "delivery": AttackStage.DELIVERY,
        "delivered": AttackStage.DELIVERY,
        "exploit": AttackStage.EXPLOITATION,
        "exploitation": AttackStage.EXPLOITATION,
        "execution": AttackStage.EXECUTION,
        "execute": AttackStage.EXECUTION,
        "executed": AttackStage.EXECUTION,
        "persistence": AttackStage.PERSISTENCE,
        "c2": AttackStage.C2,
        "command_and_control": AttackStage.C2,
        "command-and-control": AttackStage.C2,
        "lateral_movement": AttackStage.LATERAL_MOVEMENT,
        "lateral-movement": AttackStage.LATERAL_MOVEMENT,
        "collection": AttackStage.COLLECTION,
    }
    _OUTCOME_VALUES = {
        "attempt": EvidenceOutcome.ATTEMPTED,
        "attempted": EvidenceOutcome.ATTEMPTED,
        "probing": EvidenceOutcome.ATTEMPTED,
        "probe": EvidenceOutcome.ATTEMPTED,
        "fail": EvidenceOutcome.FAILED,
        "failed": EvidenceOutcome.FAILED,
        "failure": EvidenceOutcome.FAILED,
        "block": EvidenceOutcome.BLOCKED,
        "blocked": EvidenceOutcome.BLOCKED,
        "deny": EvidenceOutcome.BLOCKED,
        "denied": EvidenceOutcome.BLOCKED,
        "intercepted": EvidenceOutcome.BLOCKED,
        "deliver": EvidenceOutcome.DELIVERED,
        "delivered": EvidenceOutcome.DELIVERED,
        "landing": EvidenceOutcome.LANDED,
        "landed": EvidenceOutcome.LANDED,
        "execute": EvidenceOutcome.EXECUTED,
        "executed": EvidenceOutcome.EXECUTED,
        "execution": EvidenceOutcome.EXECUTED,
        "尝试": EvidenceOutcome.ATTEMPTED,
        "探测": EvidenceOutcome.ATTEMPTED,
        "失败": EvidenceOutcome.FAILED,
        "阻断": EvidenceOutcome.BLOCKED,
        "拦截": EvidenceOutcome.BLOCKED,
        "投递": EvidenceOutcome.DELIVERED,
        "落地": EvidenceOutcome.LANDED,
        "执行": EvidenceOutcome.EXECUTED,
    }

    def __init__(self, ndr_json: dict[str, Any] | None, llm_client: Any = None):
        """Accept graph data; ``llm_client`` is ignored for legacy call compatibility."""
        self.data: Mapping[str, Any] = ndr_json if isinstance(ndr_json, Mapping) else {}
        self._vertex_items = [item for item in self._as_list(self.data.get("vertices")) if isinstance(item, Mapping)]
        self.vertices = {
            self._text(vertex.get("id")): vertex
            for vertex in self._vertex_items
            if self._text(vertex.get("id"))
        }
        self.edges = [item for item in self._as_list(self.data.get("main_edges")) if isinstance(item, Mapping)]
        self.evidences = self._as_list(self.data.get("evidences"))

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        return list(value) if isinstance(value, (list, tuple)) else []

    @staticmethod
    def _text(value: Any, limit: int = 2048) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text[:limit]

    @classmethod
    def _safe_value(cls, value: Any, depth: int = 0) -> Any:
        """Produce JSON-compatible attributes without assuming source field shapes."""
        if depth >= 8:
            return cls._text(value, 4096)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            return {cls._text(key, 256): cls._safe_value(item, depth + 1) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._safe_value(item, depth + 1) for item in value]
        return cls._text(value, 4096)

    @classmethod
    def _parse_source_time(cls, value: Any) -> datetime | None:
        """Parse only explicit timezone-aware ISO-8601 source timestamps."""
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z") or text.endswith("z"):
            text = f"{text[:-1]}+00:00"
        match = re.match(r"^(.*\.)(\d{7,})([+-]\d{2}:?\d{2})$", text)
        if match:
            text = f"{match.group(1)}{match.group(2)[:6]}{match.group(3)}"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None

    def _entity_from_vertex(self, vertex_id: Any, fallback_role: str = "unknown") -> EntityRef | None:
        identifier = self._text(vertex_id)
        if not identifier:
            return None
        vertex = self.vertices.get(identifier, {})
        properties = vertex.get("properties") if isinstance(vertex, Mapping) else {}
        properties = properties if isinstance(properties, Mapping) else {}
        type_name = self._text(vertex.get("type") if isinstance(vertex, Mapping) else "").lower()
        kind = self._VERTEX_KINDS.get(type_name, EntityKind.UNKNOWN)
        candidates = {
            EntityKind.IP: ("ip", "address", "value"),
            EntityKind.DOMAIN: ("domain", "hostname", "value"),
            EntityKind.FILE: ("path", "file", "name", "value"),
            EntityKind.IMAGE: ("image", "process", "name", "value"),
            EntityKind.HOST: ("hostname", "host", "name", "value"),
            EntityKind.USER: ("user", "username", "name", "value"),
            EntityKind.URL: ("url", "value"),
        }.get(kind, ("value", "name"))
        value = next((self._text(properties.get(key)) for key in candidates if self._text(properties.get(key))), "")
        if not value:
            value = identifier.split(":", 1)[-1] or identifier
        role = self._text(vertex.get("role") if isinstance(vertex, Mapping) else fallback_role) or fallback_role
        attributes = {
            "vertex_id": identifier,
            "vertex_type": self._text(vertex.get("type") if isinstance(vertex, Mapping) else ""),
            "role": role,
            "properties": self._safe_value(properties),
        }
        if isinstance(vertex, Mapping):
            for key in ("is_anchor", "asset_id"):
                if key in vertex:
                    attributes[key] = self._safe_value(vertex[key])
        return EntityRef(kind=kind, value=value, attributes=attributes)

    def _edge_entities(self, edge: Mapping[str, Any]) -> list[EntityRef]:
        entities: list[EntityRef] = []
        for key, role in (("src", "source"), ("dst", "destination")):
            entity = self._entity_from_vertex(edge.get(key), role)
            if entity is not None:
                entities.append(entity)
        return entities

    def _all_entities(self) -> list[EntityRef]:
        entities: list[EntityRef] = []
        seen: set[tuple[str, str, str]] = set()
        for vertex in self._vertex_items:
            entity = self._entity_from_vertex(vertex.get("id"))
            if entity is None:
                continue
            role = self._text(entity.attributes.get("role"))
            marker = (entity.kind.value, entity.value, role)
            if marker not in seen:
                seen.add(marker)
                entities.append(entity)
        return entities

    @classmethod
    def _map_direct_outcome(cls, value: Any) -> EvidenceOutcome | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace(" ", "_")
        return cls._OUTCOME_VALUES.get(normalized) or cls._OUTCOME_VALUES.get(value.strip())

    @classmethod
    def _map_explicit_outcome_text(cls, value: Any) -> EvidenceOutcome | None:
        if not isinstance(value, str):
            return None
        text = value.strip().lower()
        if not text or re.search(r"\bnot\s+(?:blocked|denied|failed)\b|未(?:阻断|拦截|失败)", text):
            return None
        patterns = (
            (EvidenceOutcome.BLOCKED, r"\b(?:blocked|denied|intercepted)\b|(?:已)?(?:阻断|拦截)"),
            (EvidenceOutcome.FAILED, r"\b(?:failed|failure)\b|失败"),
            (EvidenceOutcome.DELIVERED, r"\bdelivered\b|(?:已)?投递"),
            (EvidenceOutcome.LANDED, r"\blanded\b|(?:已)?落地"),
            (EvidenceOutcome.EXECUTED, r"\bexecuted\b|(?:已)?执行(?:成功)?"),
            (EvidenceOutcome.ATTEMPTED, r"\b(?:attempted|attempt|probing|probe)\b|(?:已)?(?:尝试|探测)"),
        )
        for outcome, pattern in patterns:
            if re.search(pattern, text):
                return outcome
        return None

    def _outcome(self, alert_edge: Mapping[str, Any], alert: Mapping[str, Any]) -> EvidenceOutcome:
        # Status code/header/body fields are intentionally excluded from this mapping.
        for source in (alert, alert_edge):
            for key in ("attack_state", "state", "action", "result", "outcome"):
                outcome = self._map_direct_outcome(source.get(key))
                if outcome is not None:
                    return outcome
        for source in (alert, alert_edge):
            for key in ("result_text", "outcome_text", "action_text", "state_text", "message", "description", "summary", "foldedStatement"):
                outcome = self._map_explicit_outcome_text(source.get(key))
                if outcome is not None:
                    return outcome
        return EvidenceOutcome.UNKNOWN

    @classmethod
    def _map_direct_stage(cls, value: Any) -> AttackStage | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace(" ", "_")
        return cls._STAGE_VALUES.get(normalized)

    def _stage(self, alert: Mapping[str, Any]) -> AttackStage:
        for key in ("attack_stage", "stage", "mitre_stage", "tactic"):
            stage = self._map_direct_stage(alert.get(key))
            if stage is not None:
                return stage
        indicators = " ".join(
            self._text(alert.get(key)).lower()
            for key in ("threat_type", "request_body", "request_headers")
        )
        # Specific indicators precede broad exploit/reconnaissance terms.
        if re.search(r"\b(?:webshell|scheduled task|registry run key|persistence)\b|持久化|计划任务|注册表", indicators):
            return AttackStage.PERSISTENCE
        if re.search(r"\b(?:c2|command and control|beacon|tunnel)\b|命令与控制|隧道", indicators):
            return AttackStage.C2
        if re.search(r"\b(?:lateral movement|pass the hash|remote service)\b|横向移动", indicators):
            return AttackStage.LATERAL_MOVEMENT
        if re.search(r"\b(?:collection|data collection)\b|数据收集", indicators):
            return AttackStage.COLLECTION
        if re.search(r"\b(?:command execution|remote code execution|rce|exec\s*\(|curl\b|wget\b)\b|命令执行|远程代码执行", indicators):
            return AttackStage.EXECUTION
        if re.search(r"\b(?:exploit|vulnerability|injection|overflow|directory traversal)\b|漏洞|注入|溢出|目录遍历", indicators):
            return AttackStage.EXPLOITATION
        if re.search(r"\b(?:scan|scanner|crawl|recon)\b|扫描|爬虫|侦察", indicators):
            return AttackStage.RECON
        return AttackStage.UNKNOWN

    def _event_times(self) -> list[datetime]:
        """Return actual event observations, excluding later graph-distribution metadata."""
        values: list[Any] = []
        for vertex in self._vertex_items:
            properties = vertex.get("properties") if isinstance(vertex.get("properties"), Mapping) else {}
            values.extend((properties.get("first_seen_in_graph"), properties.get("last_seen_in_graph")))
        for edge in self.edges:
            values.extend((edge.get("first_seen"), edge.get("last_seen")))
            for alert_edge in self._as_list(edge.get("alert_edges")):
                if isinstance(alert_edge, Mapping):
                    values.append(alert_edge.get("ts"))
        return [parsed for value in values if (parsed := self._parse_source_time(value)) is not None]

    def _source_times(self) -> list[datetime]:
        """Include distribution time only as a legacy-display fallback."""
        event_times = self._event_times()
        distributed_at = self._parse_source_time(self.data.get("diffused_at"))
        return event_times or ([distributed_at] if distributed_at is not None else [])

    def _case_id(self) -> str:
        tenant = self._text(self.data.get("tenant"), 128) or "unknown"
        source_time = self._text(self.data.get("diffused_at"), 96)
        suffix = source_time or "graph"
        return self._text(f"NDR-{tenant}-{suffix}", 256)

    def to_case_context(self) -> CaseContext:
        """Return graph scope using event times rather than later distribution metadata."""
        times = self._event_times()
        if times:
            time_start = min(times)
            time_end = max(times)
            if time_start == time_end:
                time_end = time_start + timedelta(seconds=1)
        else:
            time_start = time_end = None
        return CaseContext(
            case_id=self._case_id(),
            tenant_id=self._text(self.data.get("tenant"), 256) or None,
            time_start=time_start,
            time_end=time_end,
            entities=self._all_entities(),
            attributes={
                "source": "NDR",
                "diffused_at": self._safe_value(self.data.get("diffused_at")),
                "vertex_count": len(self._vertex_items),
                "main_edge_count": len(self.edges),
                "graph_evidence": self._safe_value(self.evidences),
            },
        )

    def _evidence_id(self, alert_edge: Mapping[str, Any], edge_index: int, alert_index: int, seen: set[str]) -> str:
        base = self._text(alert_edge.get("alert_vid"), 220) or f"ndr-edge-{edge_index}-alert-{alert_index}"
        evidence_id = base
        sequence = 2
        while evidence_id in seen:
            evidence_id = self._text(f"{base}-{sequence}", 256)
            sequence += 1
        seen.add(evidence_id)
        return evidence_id

    def to_evidence_records(self) -> list[EvidenceRecord]:
        """Normalize every nested NDR alert edge into an immutable evidence record."""
        records: list[EvidenceRecord] = []
        seen_ids: set[str] = set()
        tenant = self._text(self.data.get("tenant"), 128) or "unknown"
        for edge_index, edge in enumerate(self.edges):
            for alert_index, alert_edge in enumerate(self._as_list(edge.get("alert_edges"))):
                if not isinstance(alert_edge, Mapping):
                    continue
                alert = alert_edge.get("alert")
                alert = alert if isinstance(alert, Mapping) else {}
                observed_at = self._parse_source_time(alert_edge.get("ts"))
                first_seen = self._parse_source_time(edge.get("first_seen"))
                last_seen = self._parse_source_time(edge.get("last_seen"))
                time_start = first_seen if first_seen is not None and last_seen is not None and first_seen < last_seen else None
                time_end = last_seen if time_start is not None else None
                alert_vid = self._text(alert_edge.get("alert_vid"), 256)
                raw_reference = self._text(f"ndr://{tenant}/{alert_vid or f'edge-{edge_index}/alert-{alert_index}'}", 4096)
                attributes = {
                    "tenant": tenant,
                    "edge": self._safe_value({
                        key: edge.get(key)
                        for key in ("src", "dst", "edge_type", "alert_count", "raw_alert_count", "occurrence_count", "occurrence_pattern", "first_seen", "last_seen")
                        if key in edge
                    }),
                    "alert_vid": alert_vid or None,
                    "folded_alert_vids": self._safe_value(self._as_list(alert_edge.get("folded_alert_vids"))),
                    "folded_statement": self._safe_value(alert_edge.get("foldedStatement")),
                    "evidence_ids": self._safe_value(self._as_list(alert_edge.get("evidence_ids"))),
                    "alert_edge": self._safe_value({key: value for key, value in alert_edge.items() if key != "alert"}),
                    "alert": self._safe_value(alert),
                }
                records.append(EvidenceRecord(
                    evidence_id=self._evidence_id(alert_edge, edge_index, alert_index, seen_ids),
                    source=EvidenceSource.NDR,
                    observed_at=observed_at or first_seen or last_seen,
                    time_start=time_start,
                    time_end=time_end,
                    entities=self._edge_entities(edge),
                    attack_stage=self._stage(alert),
                    outcome=self._outcome(alert_edge, alert),
                    confidence=1.0,
                    attributes=attributes,
                    raw_reference=raw_reference,
                ))
        return records

    def extract_entities(self) -> list[AlertEntity]:
        """Legacy entity adapter derived from the normalized graph vertices."""
        entities: list[AlertEntity] = []
        for entity in self._all_entities():
            entity_type_name = self._ALERT_ENTITY_TYPES.get(entity.kind, "HASH")
            role = self._text(entity.attributes.get("role")).lower()
            entities.append(AlertEntity(
                value=entity.value,
                type=getattr(EntityType, entity_type_name),
                role=role if role in {"attacker", "victim", "intermediate"} else "unknown",
                confidence=entity.confidence,
                context=json.dumps(entity.attributes, ensure_ascii=False, default=str),
            ))
        return entities

    def generate_atomic_facts(self, max_facts_per_edge: int = 3) -> list[str]:
        """Generate graph-derived facts without inferring results from HTTP statuses."""
        facts: list[str] = []
        for edge_index, edge in enumerate(self.edges):
            src = self._text(edge.get("src")) or "unknown"
            dst = self._text(edge.get("dst")) or "unknown"
            alerts = [item for item in self._as_list(edge.get("alert_edges")) if isinstance(item, Mapping)]
            facts.append(
                f"NDR flow {src} -> {dst}: {len(alerts)} nested alerts, "
                f"pattern={self._text(edge.get('occurrence_pattern')) or 'unknown'}"
            )
            for alert_edge in alerts[:max(0, max_facts_per_edge)]:
                alert = alert_edge.get("alert") if isinstance(alert_edge.get("alert"), Mapping) else {}
                facts.append(
                    f"NDR alert {self._text(alert_edge.get('alert_vid')) or f'edge-{edge_index}'}: "
                    f"name={self._text(alert.get('alert_name')) or 'unknown'}, "
                    f"stage={self._stage(alert).value}, outcome={self._outcome(alert_edge, alert).value}"
                )
        return facts

    def identify_information_gaps(self) -> list[str]:
        """Report only information unavailable from the parsed NDR graph."""
        gaps: list[str] = []
        if not self._vertex_items:
            gaps.append("NDR graph contains no usable vertices.")
        if not self.edges:
            gaps.append("NDR graph contains no usable main edges.")
        records = self.to_evidence_records()
        if self.edges and not records:
            gaps.append("NDR main edges contain no usable nested alert edges.")
        if records and all(record.outcome == EvidenceOutcome.UNKNOWN for record in records):
            gaps.append("NDR alerts provide no explicit outcome state, action, result, or outcome text.")
        if not self._source_times():
            gaps.append("NDR graph provides no timezone-aware source timestamp.")
        if any(not record.entities for record in records):
            gaps.append("Some NDR alerts lack usable source or destination entities.")
        return gaps

    def _infer_semantics(self) -> AlertSemantics:
        """Legacy semantic adapter based on normalized stages and source threat labels."""
        records = self.to_evidence_records()
        stages = {record.attack_stage for record in records}
        if AttackStage.RECON in stages:
            category = "reconnaissance"
        elif AttackStage.LATERAL_MOVEMENT in stages:
            category = "lateral_movement"
        elif AttackStage.PERSISTENCE in stages:
            category = "persistence"
        elif AttackStage.EXPLOITATION in stages or AttackStage.EXECUTION in stages:
            category = "intrusion"
        else:
            category = "unknown"
        severity = "medium"
        explicit_severities: list[str] = []
        tags: list[str] = []
        for edge in self.edges:
            for alert_edge in self._as_list(edge.get("alert_edges")):
                if not isinstance(alert_edge, Mapping):
                    continue
                alert = alert_edge.get("alert") if isinstance(alert_edge.get("alert"), Mapping) else {}
                severity_value = self._text(alert.get("severity")).lower()
                if severity_value in {"critical", "high", "medium", "low", "info"}:
                    explicit_severities.append(severity_value)
                for key in ("threat_type", "threat_category"):
                    value = self._text(alert.get(key))
                    if value and value not in tags:
                        tags.append(value)
        if explicit_severities:
            severity = min(explicit_severities, key=lambda value: ("critical", "high", "medium", "low", "info").index(value))
        tactic = next((record.attack_stage.value for record in records if record.attack_stage != AttackStage.UNKNOWN), None)
        return AlertSemantics(category=category, tactic=tactic, severity=severity, intent_tags=tags[:5])

    def _build_raw_summary(self) -> str:
        lines = [
            f"NDR event graph tenant={self._text(self.data.get('tenant')) or 'unknown'} "
            f"vertices={len(self._vertex_items)} main_edges={len(self.edges)}"
        ]
        lines.extend(self.generate_atomic_facts())
        return "\n".join(lines)

    def to_structured_alert(self) -> StructuredAlert:
        """Return the existing StructuredAlert contract as a data-only legacy adapter."""
        source_time = self._parse_source_time(self.data.get("diffused_at"))
        if source_time is None:
            source_times = self._source_times()
            source_time = min(source_times) if source_times else None
        return StructuredAlert(
            alert_id=self._case_id(),
            raw_alert=self._build_raw_summary(),
            timestamp=source_time,
            source_system="NDR",
            entities=self.extract_entities(),
            semantics=self._infer_semantics(),
            atomic_facts=self.generate_atomic_facts(),
            information_gaps=self.identify_information_gaps(),
            unstructured_notes=(
                f"NDR graph: {len(self._vertex_items)} vertices, {len(self.edges)} main edges, "
                f"{len(self.to_evidence_records())} normalized alert records"
            ),
        )
