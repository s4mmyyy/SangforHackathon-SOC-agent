"""Stable, database-independent evidence contracts for SOC investigations."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_QUERY_ROWS = 100_000


class EntityKind(str, Enum):
    HOST = "host"
    IP = "ip"
    PROCESS_GUID = "process_guid"
    PROCESS_ID = "process_id"
    IMAGE = "image"
    COMMAND_LINE = "command_line"
    EVENT_ID = "event_id"
    FILE = "file"
    USER = "user"
    DOMAIN = "domain"
    URL = "url"
    ALERT = "alert"
    UNKNOWN = "unknown"


class EvidenceSource(str, Enum):
    NDR = "ndr"
    EDR = "edr"
    AUDIT = "audit"
    THREAT_INTEL = "threat_intel"
    QUERY = "query"
    ANALYST = "analyst"
    UNKNOWN = "unknown"


class AttackStage(str, Enum):
    RECON = "recon"
    DELIVERY = "delivery"
    EXPLOITATION = "exploitation"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    C2 = "c2"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    UNKNOWN = "unknown"


class EvidenceOutcome(str, Enum):
    ATTEMPTED = "attempted"
    BLOCKED = "blocked"
    FAILED = "failed"
    DELIVERED = "delivered"
    LANDED = "landed"
    EXECUTED = "executed"
    COMPROMISED = "compromised"
    AUTHORIZED_NORMAL = "authorized_normal"
    UNKNOWN = "unknown"


class QueryStatus(str, Enum):
    SUCCESS = "success"
    DECLINED = "declined"
    ERROR = "error"


class LabelName(str, Enum):
    FALSE_POSITIVE = "false_positive"
    SUSPECTED_ATTACK = "suspected_attack"
    ATTACK_BLOCKED = "attack_blocked"
    ATTACK_SUCCEEDED_NOT_COMPROMISED = "attack_succeeded_not_compromised"
    COMPROMISED = "compromised"


class EntityRef(BaseModel):
    """A typed entity value that can be correlated without source-schema assumptions."""

    model_config = ConfigDict(extra="forbid")

    kind: EntityKind
    value: str = Field(min_length=1, max_length=2048)
    display_name: str | None = Field(default=None, max_length=2048)
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

    @field_validator("value")
    @classmethod
    def strip_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("entity value must not be empty")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be a finite value from 0 to 1")
        return value


class CaseContext(BaseModel):
    """Scope and correlation boundaries for a single investigation case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=256)
    tenant_id: str | None = Field(default=None, max_length=256)
    time_start: datetime | None = None
    time_end: datetime | None = None
    entities: list[EntityRef] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("time_start", "time_end", "created_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "CaseContext":
        if (self.time_start is None) != (self.time_end is None):
            raise ValueError("time_start and time_end must be supplied together")
        if self.time_start is not None and self.time_start >= self.time_end:
            raise ValueError("time_start must be before time_end")
        return self


class EvidenceRecord(BaseModel):
    """An immutable-style normalized observation; raw data remains externally referenced."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=256)
    source: EvidenceSource = EvidenceSource.UNKNOWN
    observed_at: datetime | None = None
    time_start: datetime | None = None
    time_end: datetime | None = None
    entities: list[EntityRef] = Field(default_factory=list)
    attack_stage: AttackStage = AttackStage.UNKNOWN
    outcome: EvidenceOutcome = EvidenceOutcome.UNKNOWN
    confidence: float = 0.5
    attributes: dict[str, Any] = Field(default_factory=dict)
    raw_reference: str | None = Field(default=None, max_length=4096)
    query_id: str | None = Field(default=None, max_length=256)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("observed_at", "time_start", "time_end", "recorded_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must include a timezone")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be a finite value from 0 to 1")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "EvidenceRecord":
        if (self.time_start is None) != (self.time_end is None):
            raise ValueError("evidence time_start and time_end must be supplied together")
        if self.time_start is not None and self.time_start >= self.time_end:
            raise ValueError("evidence time_start must be before time_end")
        return self


class InvestigationGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gap_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=256)
    code: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=2048)
    reason: str = Field(min_length=1, max_length=2048)
    target_entities: list[EntityRef] = Field(default_factory=list)
    time_start: datetime | None = None
    time_end: datetime | None = None
    priority: int = Field(default=50, ge=1, le=100)

    @field_validator("time_start", "time_end")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "InvestigationGap":
        if (self.time_start is None) != (self.time_end is None):
            raise ValueError("gap time_start and time_end must be supplied together")
        if self.time_start is not None and self.time_start >= self.time_end:
            raise ValueError("gap time_start must be before time_end")
        return self


class InvestigationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=256)
    objective: str = Field(min_length=1, max_length=128)
    target_entities: list[EntityRef] = Field(default_factory=list)
    data_source: EvidenceSource = EvidenceSource.QUERY
    time_start: datetime
    time_end: datetime
    expected_discrimination: str = Field(min_length=1, max_length=2048)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    max_rows: int = Field(default=1_000, ge=1, le=MAX_QUERY_ROWS)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("time_start", "time_end")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @field_validator("estimated_cost")
    @classmethod
    def finite_cost(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("estimated_cost must be finite")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "InvestigationTask":
        if self.time_start >= self.time_end:
            raise ValueError("time_start must be before time_end")
        return self


class AuditTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=256)
    event: str = Field(min_length=1, max_length=128)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool | None = None
    query_id: str | None = Field(default=None, max_length=256)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("recorded_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=256)
    status: QueryStatus
    reason: str | None = Field(default=None, max_length=4096)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0, le=MAX_QUERY_ROWS)
    sql: str | None = Field(default=None, max_length=16384)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audit: list[AuditTrace] = Field(default_factory=list)

    @field_validator("started_at", "finished_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> "QueryResult":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.row_count != len(self.rows):
            raise ValueError("row_count must equal the number of rows")
        return self


class LabelDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: int = Field(ge=1, le=5)
    label_name: LabelName
    confidence: float
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradictory_evidence_ids: list[str] = Field(default_factory=list)
    information_gaps: list[InvestigationGap] = Field(default_factory=list)
    why_not_higher: str = Field(min_length=1, max_length=4096)
    why_not_lower: str = Field(min_length=1, max_length=4096)
    rationale: str = Field(min_length=1, max_length=4096)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be a finite value from 0 to 1")
        return value

    @field_validator("decided_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_label_name(self) -> "LabelDecision":
        expected = {
            1: LabelName.FALSE_POSITIVE,
            2: LabelName.SUSPECTED_ATTACK,
            3: LabelName.ATTACK_BLOCKED,
            4: LabelName.ATTACK_SUCCEEDED_NOT_COMPROMISED,
            5: LabelName.COMPROMISED,
        }[self.label]
        if self.label_name != expected:
            raise ValueError("label_name must match label")
        return self
