"""Deterministic, evidence-threshold label policy with no LLM dependency."""

from __future__ import annotations

from collections.abc import Iterable

from evidence_models import (
    CaseContext,
    EntityRef,
    EvidenceOutcome,
    EvidenceRecord,
    InvestigationGap,
    LabelDecision,
    LabelName,
)


_LABELS: dict[int, LabelName] = {
    1: LabelName.FALSE_POSITIVE,
    2: LabelName.SUSPECTED_ATTACK,
    3: LabelName.ATTACK_BLOCKED,
    4: LabelName.ATTACK_SUCCEEDED_NOT_COMPROMISED,
    5: LabelName.COMPROMISED,
}

# Attribute keys are deliberately allowlisted. Values such as HTTP status are not read.
_VETTED_BOOLEAN_ATTRIBUTES = frozenset(
    {
        "compromise_confirmed",
        "command_execution_confirmed",
        "webshell_execution_confirmed",
        "persistence_confirmed",
        "credential_access_confirmed",
        "c2_confirmed",
        "lateral_movement_confirmed",
        "data_access_confirmed",
        "payload_delivered_confirmed",
        "file_landed_confirmed",
        "file_upload_confirmed",
        "security_control_blocked",
        "authorized_normal_activity",
        "approved_scan",
        "benign_explanation_confirmed",
        "attack_indicator_confirmed",
    }
)


class LabelPolicy:
    """Choose one label only from explicit outcomes and vetted structured facts."""

    def decide(
        self, case: CaseContext, evidence: Iterable[EvidenceRecord]
    ) -> LabelDecision:
        records = list(evidence)
        compromise = self._matching(records, self._is_compromise)
        landed = self._matching(records, self._is_landed_or_delivered)
        blocked = self._matching(records, self._is_explicitly_blocked)
        attack = self._matching(records, self._is_attack_indicator)
        normal = self._matching(records, self._is_authorized_normal)

        if compromise:
            return self._decision_for_compromise(case, compromise, landed, blocked, normal)
        if landed:
            return self._decision_for_landed(case, landed, blocked, normal)
        if blocked and (
            attack
            or any(record.outcome == EvidenceOutcome.BLOCKED for record in blocked)
        ):
            return self._decision_for_blocked(case, blocked, attack, normal)
        if attack:
            return self._decision_for_suspected(case, attack, normal, empty=False)
        if normal:
            return self._decision_for_false_positive(case, normal)
        return self._decision_for_suspected(case, [], [], empty=True)

    @staticmethod
    def _matching(
        records: list[EvidenceRecord], predicate: object
    ) -> list[EvidenceRecord]:
        return [record for record in records if predicate(record)]  # type: ignore[operator]

    @staticmethod
    def _vetted(record: EvidenceRecord, key: str) -> bool:
        return key in _VETTED_BOOLEAN_ATTRIBUTES and record.attributes.get(key) is True

    def _is_compromise(self, record: EvidenceRecord) -> bool:
        return record.outcome in {EvidenceOutcome.EXECUTED, EvidenceOutcome.COMPROMISED} or any(
            self._vetted(record, key)
            for key in (
                "compromise_confirmed",
                "command_execution_confirmed",
                "webshell_execution_confirmed",
                "persistence_confirmed",
                "credential_access_confirmed",
                "c2_confirmed",
                "lateral_movement_confirmed",
                "data_access_confirmed",
            )
        )

    def _is_landed_or_delivered(self, record: EvidenceRecord) -> bool:
        return record.outcome in {EvidenceOutcome.DELIVERED, EvidenceOutcome.LANDED} or any(
            self._vetted(record, key)
            for key in (
                "payload_delivered_confirmed",
                "file_landed_confirmed",
                "file_upload_confirmed",
            )
        )

    def _is_explicitly_blocked(self, record: EvidenceRecord) -> bool:
        return record.outcome == EvidenceOutcome.BLOCKED or self._vetted(
            record, "security_control_blocked"
        )

    def _is_attack_indicator(self, record: EvidenceRecord) -> bool:
        return record.outcome == EvidenceOutcome.ATTEMPTED or self._vetted(
            record, "attack_indicator_confirmed"
        )

    def _is_authorized_normal(self, record: EvidenceRecord) -> bool:
        return record.outcome == EvidenceOutcome.AUTHORIZED_NORMAL or any(
            self._vetted(record, key)
            for key in (
                "authorized_normal_activity",
                "approved_scan",
                "benign_explanation_confirmed",
            )
        )

    @staticmethod
    def _confidence(records: list[EvidenceRecord], fallback: float) -> float:
        if not records:
            return fallback
        return round(sum(record.confidence for record in records) / len(records), 3)

    @staticmethod
    def _ids(records: list[EvidenceRecord]) -> list[str]:
        return [record.evidence_id for record in records]

    @staticmethod
    def _gap(
        case: CaseContext,
        code: str,
        description: str,
        reason: str,
        priority: int,
    ) -> InvestigationGap:
        return InvestigationGap(
            code=code,
            description=description,
            reason=reason,
            target_entities=case.entities,
            time_start=case.time_start,
            time_end=case.time_end,
            priority=priority,
        )

    def _decision_for_compromise(
        self,
        case: CaseContext,
        compromise: list[EvidenceRecord],
        landed: list[EvidenceRecord],
        blocked: list[EvidenceRecord],
        normal: list[EvidenceRecord],
    ) -> LabelDecision:
        contradictions = self._ids(normal + blocked)
        return LabelDecision(
            label=5,
            label_name=_LABELS[5],
            confidence=self._confidence(compromise, 0.5),
            supporting_evidence_ids=self._ids(compromise),
            contradictory_evidence_ids=contradictions,
            information_gaps=[
                self._gap(
                    case,
                    "scope_and_impact",
                    "Confirm the scope, duration, and impact of the confirmed compromise.",
                    "Compromise behavior is explicit, but impact scope may remain unknown.",
                    70,
                )
            ],
            why_not_higher="Label 5 is the highest label in this policy.",
            why_not_lower="Explicit structured compromise behavior is present; delivery or blocking alone cannot lower the label.",
            rationale="Explicit structured evidence confirms compromise behavior.",
        )

    def _decision_for_landed(
        self,
        case: CaseContext,
        landed: list[EvidenceRecord],
        blocked: list[EvidenceRecord],
        normal: list[EvidenceRecord],
    ) -> LabelDecision:
        return LabelDecision(
            label=4,
            label_name=_LABELS[4],
            confidence=self._confidence(landed, 0.5),
            supporting_evidence_ids=self._ids(landed),
            contradictory_evidence_ids=self._ids(blocked + normal),
            information_gaps=[
                self._gap(
                    case,
                    "post_delivery_execution",
                    "Check for command execution, persistence, C2, lateral movement, credential access, or data access after delivery.",
                    "Delivery or file landing is explicit, but no explicit compromise behavior is available.",
                    95,
                )
            ],
            why_not_higher="No explicit structured evidence confirms execution, persistence, C2, lateral movement, credential access, data access, or compromise.",
            why_not_lower="Explicit structured payload delivery or file landing exceeds an unconfirmed attempt or control block.",
            rationale="Explicit structured evidence confirms payload delivery or file landing without confirmed compromise behavior.",
        )

    def _decision_for_blocked(
        self,
        case: CaseContext,
        blocked: list[EvidenceRecord],
        attack: list[EvidenceRecord],
        normal: list[EvidenceRecord],
    ) -> LabelDecision:
        return LabelDecision(
            label=3,
            label_name=_LABELS[3],
            confidence=self._confidence(blocked, 0.5),
            supporting_evidence_ids=self._ids(blocked),
            contradictory_evidence_ids=self._ids(normal),
            information_gaps=[
                self._gap(
                    case,
                    "block_effectiveness",
                    "Confirm whether any payload was delivered or landed despite the security-control block.",
                    "An explicit block is present, but downstream delivery and host activity remain unverified.",
                    85,
                )
            ],
            why_not_higher="No explicit structured delivery, landing, or compromise outcome is present.",
            why_not_lower="An explicit structured security-control block is stronger than an unconfirmed attack indicator.",
            rationale="Explicit structured evidence records a security-control block.",
        )

    def _decision_for_suspected(
        self,
        case: CaseContext,
        attack: list[EvidenceRecord],
        normal: list[EvidenceRecord],
        *,
        empty: bool,
    ) -> LabelDecision:
        if empty:
            rationale = "No evidence is available; the conservative default is suspected_attack pending validation."
            confidence = 0.1
            support: list[str] = []
            gap_reason = "No evidence is available to verify whether the case is benign, blocked, delivered, or compromised."
        else:
            rationale = "Structured attack indicators are present without explicit block, delivery, landing, or compromise outcomes."
            confidence = self._confidence(attack, 0.35)
            support = self._ids(attack)
            gap_reason = "Attack indicators exist, but decisive control, delivery, and endpoint behavior evidence is absent."
        return LabelDecision(
            label=2,
            label_name=_LABELS[2],
            confidence=confidence,
            supporting_evidence_ids=support,
            contradictory_evidence_ids=self._ids(normal),
            information_gaps=[
                self._gap(
                    case,
                    "decisive_outcome",
                    "Collect explicit security-control action and endpoint/application outcome evidence for the case time window.",
                    gap_reason,
                    95,
                ),
                self._gap(
                    case,
                    "benign_context",
                    "Validate authorization, allowlisting, or a credible normal-business explanation.",
                    "No credible authorized-normal evidence is available to support false_positive.",
                    60,
                ),
            ],
            why_not_higher="No explicit structured security-control block, delivery, landing, or compromise outcome is present.",
            why_not_lower="No credible authorized-normal activity, approved scan, or benign explanation is explicitly confirmed.",
            rationale=rationale,
        )

    def _decision_for_false_positive(
        self, case: CaseContext, normal: list[EvidenceRecord]
    ) -> LabelDecision:
        return LabelDecision(
            label=1,
            label_name=_LABELS[1],
            confidence=self._confidence(normal, 0.5),
            supporting_evidence_ids=self._ids(normal),
            contradictory_evidence_ids=[],
            information_gaps=[
                self._gap(
                    case,
                    "attack_evidence_review",
                    "Retain enough telemetry to detect later credible attack indicators in the same scope.",
                    "Current structured evidence supports a normal or authorized explanation only.",
                    30,
                )
            ],
            why_not_higher="No explicit structured attack indicator, control block, delivery, landing, or compromise outcome is present.",
            why_not_lower="Label 1 is the lowest label in this policy.",
            rationale="Credible structured evidence explicitly identifies the activity as authorized or normal, with no credible attack evidence.",
        )
