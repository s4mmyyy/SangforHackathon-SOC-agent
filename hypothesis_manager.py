"""假设管理与结构化证据账本。

最终标签只由 ``LabelPolicy`` 根据结构化证据决定。贝叶斯分数和可选 LLM
仅用于兼容旧调用方的调查排序与解释，绝不会覆盖策略标签。
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from evidence_models import (
    CaseContext,
    EntityKind,
    EntityRef,
    EvidenceOutcome,
    EvidenceRecord,
    EvidenceSource,
    InvestigationTask,
    LabelDecision,
)
from label_policy import LabelPolicy


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    PENDING = "pending"


class EvidenceType(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


class HypothesisCategory(str, Enum):
    FALSE_POSITIVE = "false_positive"
    SUSPECTED_ATTACK = "suspected_attack"
    ATTACK_BLOCKED = "attack_blocked"
    ATTACK_SUCCEEDED_NOT_COMPROMISED = "attack_succeeded_not_compromised"
    COMPROMISED = "compromised"


LABEL_TO_CATEGORY = {
    1: HypothesisCategory.FALSE_POSITIVE,
    2: HypothesisCategory.SUSPECTED_ATTACK,
    3: HypothesisCategory.ATTACK_BLOCKED,
    4: HypothesisCategory.ATTACK_SUCCEEDED_NOT_COMPROMISED,
    5: HypothesisCategory.COMPROMISED,
}


@dataclass
class Evidence:
    """旧贝叶斯接口的证据视图；不替代结构化 ``EvidenceRecord``。"""

    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = ""
    raw_content: str = ""
    related_entities: List[str] = field(default_factory=list)
    evidence_type: EvidenceType = EvidenceType.NEUTRAL
    likelihood_ratio: float = 1.0
    weight: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reasoning: str = ""


@dataclass
class Hypothesis:
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    category: str = ""
    prior_probability: float = 0.5
    posterior_probability: float = 0.5
    evidences: List[Evidence] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    expected_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conclusion_reasoning: str = ""


@dataclass
class InvestigationRecommendation:
    priority: Literal["critical", "high", "medium", "low"]
    action: str = ""
    target_entities: List[str] = field(default_factory=list)
    rationale: str = ""
    expected_outcome: str = ""


class BayesianEngine:
    """保留旧接口的有限贝叶斯排序器，不用于最终标签。"""

    @staticmethod
    def probability_to_log_odds(probability: float) -> float:
        probability = max(0.0001, min(0.9999, probability))
        return math.log(probability / (1 - probability))

    @staticmethod
    def log_odds_to_probability(log_odds: float) -> float:
        return 1.0 / (1.0 + math.exp(-log_odds))

    @staticmethod
    def likelihood_ration_to_log_weight(likelihood_ratio: float, weight: float = 1.0) -> float:
        return max(0.0, min(1.0, weight)) * math.log(max(0.01, min(100.0, likelihood_ratio)))

    def update(self, hypothesis: Hypothesis, evidence: Evidence) -> float:
        if hypothesis.status != HypothesisStatus.ACTIVE:
            return hypothesis.posterior_probability
        old = self.probability_to_log_odds(hypothesis.posterior_probability)
        new = self.log_odds_to_probability(old + self.likelihood_ration_to_log_weight(
            evidence.likelihood_ratio, evidence.weight
        ))
        hypothesis.posterior_probability = new
        hypothesis.evidences.append(evidence)
        hypothesis.updated_at = datetime.now(timezone.utc)
        return new

    def batch_update(self, hypothesis: Hypothesis, evidences: List[Evidence]) -> float:
        for evidence in evidences:
            self.update(hypothesis, evidence)
        return hypothesis.posterior_probability


class LikelihoodRatioEstimator:
    """可选 LLM 估计器；任何故障都会保守回退到确定性规则。"""

    QUICK_PATTERNS = {
        "false_positive_indicators": ("误报", "false positive", "正常业务", "白名单", "whitelist", "approved"),
        "blocked_indicators": ("blocked", "拦截", "waf", "ips", "security control"),
        "delivery_indicators": ("uploaded", "file landed", "文件落地", "payload delivered"),
        "compromise_indicators": ("webshell", "command execution", "命令执行", "c2", "beacon", "横向移动"),
        "attack_indicators": ("scan", "扫描", "probe", "探测", "exploit", "injection", "payload"),
    }

    def __init__(self, llm_client: Any = None) -> None:
        self.llm = llm_client
        self.consecutive_failures = 0
        self.max_failures = 3
        self.circuit_open = False

    def estimate(
        self,
        evidence_content: str,
        hypothesis_category: str,
        hypothesis_name: str = "",
        hypothesis_desc: str = "",
    ) -> Tuple[float, str]:
        content = evidence_content or ""
        lowered = content.lower()
        tags = [name for name, words in self.QUICK_PATTERNS.items() if any(word in lowered for word in words)]
        if not self.llm or self.circuit_open:
            return self._rule_fallback(content, hypothesis_category, tags)
        try:
            result = self._llm_estimate(content, hypothesis_category, hypothesis_name, hypothesis_desc, tags)
        except Exception:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self.circuit_open = True
            return self._rule_fallback(content, hypothesis_category, tags)
        self.consecutive_failures = 0
        return result

    def _llm_estimate(
        self,
        evidence_content: str,
        hypothesis_category: str,
        hypothesis_name: str,
        hypothesis_desc: str,
        matched_tags: List[str],
    ) -> Tuple[float, str]:
        if not hasattr(self.llm, "chat"):
            raise TypeError("llm client must expose chat")
        prompt = (
            "请仅输出 JSON：{\"likelihood_ratio\": 0.01-100, \"reasoning\": \"...\"}。\n"
            f"假设类别：{hypothesis_category}\n假设：{hypothesis_name}\n描述：{hypothesis_desc}\n"
            f"规则线索：{matched_tags}\n证据：{evidence_content[:4000]}"
        )
        response = self.llm.chat("你是安全证据评估助手，只评估调查优先级，不裁决标签。", prompt)
        if not isinstance(response, str):
            raise TypeError("llm response must be text")
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        payload = json.loads(text)
        likelihood_ratio = float(payload["likelihood_ratio"])
        if not math.isfinite(likelihood_ratio):
            raise ValueError("invalid likelihood ratio")
        reasoning = str(payload.get("reasoning", "LLM 调查优先级评估"))
        return max(0.01, min(100.0, likelihood_ratio)), reasoning

    def _rule_fallback(
        self, evidence_content: str, hypothesis_category: str, matched_tags: List[str]
    ) -> Tuple[float, str]:
        tags = set(matched_tags)
        category = str(hypothesis_category)
        if category == HypothesisCategory.FALSE_POSITIVE.value:
            lr = 3.0 if "false_positive_indicators" in tags else 0.7 if tags else 1.0
        elif category == HypothesisCategory.SUSPECTED_ATTACK.value:
            lr = 2.0 if "attack_indicators" in tags else 1.2 if tags else 1.0
        elif category == HypothesisCategory.ATTACK_BLOCKED.value:
            lr = 4.0 if "blocked_indicators" in tags else 0.8 if "compromise_indicators" in tags else 1.0
        elif category == HypothesisCategory.ATTACK_SUCCEEDED_NOT_COMPROMISED.value:
            lr = 4.0 if "delivery_indicators" in tags else 0.7 if "blocked_indicators" in tags else 1.0
        elif category == HypothesisCategory.COMPROMISED.value:
            lr = 5.0 if "compromise_indicators" in tags else 0.6 if "blocked_indicators" in tags else 1.0
        else:
            lr = 1.0
        return lr, f"规则降级评估（仅用于调查排序）：{', '.join(matched_tags) or '无明确线索'}"


class LLMResponseAnalyzer:
    """保留旧名称。响应状态码不被解释为阻断、投递或失陷事实。"""

    def __init__(self, llm_client: Any = None) -> None:
        self.llm = llm_client

    def analyze(self, request_headers: str, request_body: str, response_headers: str, response_body: str, alert_name: str) -> Dict[str, Any]:
        del request_headers, request_body, response_headers, response_body, alert_name
        return {
            "result": "suspicious",
            "confidence": 0.0,
            "reasoning": "HTTP 交互需要显式安全控制或端点证据；状态码不足以确认攻击结果。",
        }


class HypothesisManager:
    """兼容旧假设接口，并提供以结构化证据账本为主的研判入口。"""

    CONFIRMATION_THRESHOLD = 0.85
    REJECTION_THRESHOLD = 0.15
    DEEP_INVESTIGATION_THRESHOLD = 0.40
    MAX_TASKS = 3

    def __init__(self, llm_client: Any = None) -> None:
        self.llm = llm_client
        self.bayesian = BayesianEngine()
        self.lr_estimator = LikelihoodRatioEstimator(llm_client)
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.investigation_history: List[Dict[str, Any]] = []
        self.policy = LabelPolicy()
        self.case_context: CaseContext | None = None
        self._ledger: Dict[str, EvidenceRecord] = {}
        self._label_decision: LabelDecision | None = None
        self.judge_engine = LLMJudgeEngine(self.policy)

    def _create_hypothesis(self, name: str, description: str, category: str, prior: float, expected_evidence: List[str] | None = None) -> Hypothesis:
        hypothesis = Hypothesis(
            name=name,
            description=description,
            category=str(category),
            prior_probability=prior,
            posterior_probability=prior,
            expected_evidence=expected_evidence or [],
        )
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        return hypothesis

    def _initialize_hypotheses(self) -> None:
        self.hypotheses.clear()
        definitions = (
            ("误报/正常业务", "存在可验证的授权或正常解释", HypothesisCategory.FALSE_POSITIVE, 0.15),
            ("疑似攻击", "存在攻击指标但没有决定性结果", HypothesisCategory.SUSPECTED_ATTACK, 0.35),
            ("攻击被拦截", "存在明确安全控制阻断", HypothesisCategory.ATTACK_BLOCKED, 0.20),
            ("攻击成功未失陷", "存在明确投递或文件落地", HypothesisCategory.ATTACK_SUCCEEDED_NOT_COMPROMISED, 0.15),
            ("主机已失陷", "存在明确执行、C2、持久化或横向移动", HypothesisCategory.COMPROMISED, 0.15),
        )
        for name, description, category, prior in definitions:
            self._create_hypothesis(name, description, category.value, prior)

    def initialize_from_evidence(
        self, case_context: CaseContext, evidence_records: Iterable[EvidenceRecord]
    ) -> List[Hypothesis]:
        """以账本初始化案件；唯一标签来自 ``LabelPolicy``。"""
        if not isinstance(case_context, CaseContext):
            case_context = CaseContext.model_validate(case_context)
        self.case_context = case_context
        self._ledger.clear()
        self._label_decision = None
        self.investigation_history.clear()
        self._initialize_hypotheses()
        self.add_structured_evidence(evidence_records, refresh=False)
        self._refresh_label_decision()
        return list(self.hypotheses.values())

    def add_structured_evidence(self, records: Iterable[EvidenceRecord], *, refresh: bool = True) -> LabelDecision | None:
        """追加去重的结构化事实，随后重新运行确定性标签策略。"""
        new_records: list[EvidenceRecord] = []
        for record in records:
            if not isinstance(record, EvidenceRecord):
                record = EvidenceRecord.model_validate(record)
            if record.evidence_id in self._ledger:
                continue
            self._ledger[record.evidence_id] = record
            new_records.append(record)
        for record in new_records:
            self._add_legacy_view(record)
        if refresh:
            self._refresh_label_decision()
        return self._label_decision

    def _add_legacy_view(self, record: EvidenceRecord) -> None:
        text = self._record_summary(record)
        entities = [entity.value for entity in record.entities]
        for hypothesis in self.hypotheses.values():
            lr, reasoning = self.lr_estimator._rule_fallback(text, hypothesis.category, [])
            view = Evidence(
                evidence_id=f"{record.evidence_id}:{hypothesis.hypothesis_id}",
                source=record.source.value,
                raw_content=text,
                related_entities=list(entities),
                evidence_type=self._evidence_type_for_lr(lr),
                likelihood_ratio=lr,
                weight=min(record.confidence, 0.8),
                timestamp=record.observed_at or record.recorded_at,
                reasoning=reasoning,
            )
            self.bayesian.update(hypothesis, view)

    @staticmethod
    def _record_summary(record: EvidenceRecord) -> str:
        entities = ", ".join(entity.value for entity in record.entities[:4])
        return f"source={record.source.value}; outcome={record.outcome.value}; stage={record.attack_stage.value}; entities={entities}"

    def _refresh_label_decision(self) -> None:
        if self.case_context is not None:
            self._label_decision = self.policy.decide(self.case_context, self._ledger.values())

    def get_label_decision(self) -> LabelDecision | None:
        return self._label_decision

    @property
    def evidence_ledger(self) -> List[EvidenceRecord]:
        return list(self._ledger.values())

    def create_investigation_tasks(self) -> List[InvestigationTask]:
        """把策略信息缺口转成有边界的查询任务；无有效范围时绝不生成任务。"""
        case, decision = self.case_context, self._label_decision
        if case is None or decision is None or not case.entities or case.time_start is None or case.time_end is None:
            return []
        if case.time_start >= case.time_end:
            return []
        tasks: list[InvestigationTask] = []
        seen: set[str] = set()
        for gap in decision.information_gaps:
            objective = "security_control_investigation" if gap.code in {"decisive_outcome", "block_effectiveness"} else "process_investigation"
            key = f"{objective}:{gap.code}"
            if key in seen:
                continue
            seen.add(key)
            tasks.append(InvestigationTask(
                objective=objective,
                target_entities=gap.target_entities or case.entities,
                data_source=EvidenceSource.QUERY,
                time_start=case.time_start,
                time_end=case.time_end,
                expected_discrimination=f"补充“{gap.description}”，用于区分当前标签及相邻标签。",
                estimated_cost=round(min(1.0, gap.priority / 100.0), 2),
                max_rows=1000,
                attributes={"gap_code": gap.code, "case_id": case.case_id},
            ))
            if len(tasks) >= self.MAX_TASKS:
                break
        return tasks

    def initialize_from_alert(self, structured_alert: Any) -> List[Hypothesis]:
        """旧入口：从 ``StructuredAlert`` 仅导出保守的结构化事实。"""
        timestamp = self._as_aware(getattr(structured_alert, "timestamp", None))
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        entities = [self._legacy_entity(entity) for entity in getattr(structured_alert, "entities", [])]
        entities = [entity for entity in entities if entity is not None]
        case = CaseContext(
            case_id=str(getattr(structured_alert, "alert_id", "legacy-alert")),
            time_start=timestamp,
            time_end=timestamp + timedelta(seconds=1),
            entities=entities,
            attributes={"source_system": str(getattr(structured_alert, "source_system", "legacy"))},
        )
        records: list[EvidenceRecord] = []
        for index, fact in enumerate(getattr(structured_alert, "atomic_facts", []) or []):
            text = str(fact)
            lowered = text.lower()
            outcome = EvidenceOutcome.UNKNOWN
            attributes: dict[str, Any] = {}
            if any(token in lowered for token in ("authorized", "正常业务", "白名单", "批准扫描")):
                outcome = EvidenceOutcome.AUTHORIZED_NORMAL
                attributes["authorized_normal_activity"] = True
            elif any(token in lowered for token in ("明确拦截", "security_control_blocked", "安全控制阻断")):
                outcome = EvidenceOutcome.BLOCKED
                attributes["security_control_blocked"] = True
            elif any(token in lowered for token in ("攻击", "attack", "扫描", "探测", "payload", "注入")):
                outcome = EvidenceOutcome.ATTEMPTED
                attributes["attack_indicator_confirmed"] = True
            records.append(EvidenceRecord(
                evidence_id=f"{case.case_id}:legacy:{index}",
                source=EvidenceSource.UNKNOWN,
                observed_at=timestamp,
                entities=entities,
                outcome=outcome,
                confidence=0.4,
                attributes=attributes,
                raw_reference=f"legacy_atomic_fact:{index}",
            ))
        return self.initialize_from_evidence(case, records)

    @staticmethod
    def _as_aware(value: Any) -> datetime | None:
        if not isinstance(value, datetime):
            return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None or value.utcoffset() is None else value

    @staticmethod
    def _legacy_entity(entity: Any) -> EntityRef | None:
        value = getattr(entity, "value", None)
        if not isinstance(value, str) or not value.strip():
            return None
        raw_kind = str(getattr(entity, "type", "unknown")).lower()
        aliases = {"hostname": "host", "process": "process_guid", "hash": "unknown", "port": "unknown"}
        kind = aliases.get(raw_kind, raw_kind)
        return EntityRef(kind=EntityKind(kind) if kind in EntityKind._value2member_map_ else EntityKind.UNKNOWN, value=value, confidence=float(getattr(entity, "confidence", 0.5)))

    def add_evidence(self, evidence: Evidence) -> Dict[str, float]:
        """旧接口：每个假设写入独立证据副本，不修改调用方对象。"""
        results: Dict[str, float] = {}
        for hypothesis in self.hypotheses.values():
            if hypothesis.status != HypothesisStatus.ACTIVE:
                continue
            lr, reasoning = self.lr_estimator.estimate(
                evidence.raw_content, hypothesis.category, hypothesis.name, hypothesis.description
            )
            per_hypothesis = copy.copy(evidence)
            per_hypothesis.evidence_id = f"{evidence.evidence_id}:{hypothesis.hypothesis_id}"
            per_hypothesis.related_entities = list(evidence.related_entities)
            per_hypothesis.likelihood_ratio = lr
            per_hypothesis.reasoning = reasoning
            per_hypothesis.evidence_type = self._evidence_type_for_lr(lr)
            results[hypothesis.hypothesis_id] = self.bayesian.update(hypothesis, per_hypothesis)
            self._evaluate_hypothesis_status(hypothesis)
        self.investigation_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence": evidence.raw_content[:100],
            "probabilities": {key: value.posterior_probability for key, value in self.hypotheses.items()},
        })
        return results

    @staticmethod
    def _evidence_type_for_lr(likelihood_ratio: float) -> EvidenceType:
        if likelihood_ratio > 1.5:
            return EvidenceType.SUPPORTING
        if likelihood_ratio < 0.7:
            return EvidenceType.CONTRADICTING
        return EvidenceType.NEUTRAL

    def _evaluate_hypothesis_status(self, hypothesis: Hypothesis) -> None:
        probability = hypothesis.posterior_probability
        if probability >= self.CONFIRMATION_THRESHOLD:
            hypothesis.status = HypothesisStatus.CONFIRMED
            hypothesis.conclusion_reasoning = "该假设的调查排序分数达到确认阈值；最终标签仍必须以 LabelPolicy 为准。"
        elif probability <= self.REJECTION_THRESHOLD:
            hypothesis.status = HypothesisStatus.REJECTED
            hypothesis.conclusion_reasoning = "该假设的调查排序分数低于阈值；最终标签仍必须以 LabelPolicy 为准。"

    def get_top_hypothesis(self) -> Optional[Hypothesis]:
        active = [item for item in self.hypotheses.values() if item.status == HypothesisStatus.ACTIVE]
        return max(active, key=lambda item: item.posterior_probability) if active else None

    def _calculate_entropy(self) -> float:
        active = [item.posterior_probability for item in self.hypotheses.values() if item.status == HypothesisStatus.ACTIVE]
        total = sum(active)
        return -sum((item / total) * math.log2(item / total) for item in active if item > 0) if total else 0.0

    def get_status(self) -> Dict[str, Any]:
        active = [item for item in self.hypotheses.values() if item.status == HypothesisStatus.ACTIVE]
        confirmed = [item for item in self.hypotheses.values() if item.status == HypothesisStatus.CONFIRMED]
        rejected = [item for item in self.hypotheses.values() if item.status == HypothesisStatus.REJECTED]
        return {
            "has_conclusion": self._label_decision is not None,
            "label_decision": self._label_decision,
            "confirmed_hypotheses": confirmed,
            "active_hypotheses": active,
            "rejected_hypotheses": rejected,
            "stalemate": False,
            "top_hypothesis": self.get_top_hypothesis(),
            "entropy": self._calculate_entropy(),
        }

    def generate_investigation_recommendations(self) -> List[InvestigationRecommendation]:
        tasks = self.create_investigation_tasks()
        if tasks:
            return [InvestigationRecommendation(
                priority="high",
                action=f"执行受限调查：{task.objective}",
                target_entities=[entity.value for entity in task.target_entities],
                rationale=task.expected_discrimination,
                expected_outcome="获取可改变策略标签的显式结构化证据。",
            ) for task in tasks]
        decision = self._label_decision
        if decision is not None:
            return [InvestigationRecommendation(
                priority="medium",
                action="保留当前策略结论并等待可关联证据",
                rationale=decision.rationale,
                expected_outcome="不在无实体或无时间范围时执行无界查询。",
            )]
        return []

    def generate_adversarial_analysis(self, target_hypothesis_id: str | None = None) -> str:
        del target_hypothesis_id
        if self._label_decision is None:
            return "尚无结构化标签决策；需要补充可审计事实。"
        gaps = "；".join(gap.description for gap in self._label_decision.information_gaps)
        return f"当前标签为 {self._label_decision.label_name.value}。需重点反证或补充：{gaps or '无'}"

    def generate_report(self) -> Dict[str, Any]:
        status = self.get_status()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label_decision": self._label_decision.model_dump(mode="json") if self._label_decision else None,
            "evidence_ledger_count": len(self._ledger),
            "summary": {"total_hypotheses": len(self.hypotheses), "uncertainty_entropy": round(status["entropy"], 4)},
            "hypotheses": [{"id": item.hypothesis_id, "name": item.name, "category": item.category, "posterior": round(item.posterior_probability, 4)} for item in self.hypotheses.values()],
            "top_recommendations": [item.__dict__ for item in self.generate_investigation_recommendations()],
            "adversarial_analysis": self.generate_adversarial_analysis(),
            "investigation_history": self.investigation_history[-5:],
        }


class LLMJudgeEngine:
    """旧裁决器名称的兼容解释器：不调用 LLM，且不能覆盖 LabelPolicy。"""

    def __init__(self, policy: LabelPolicy | Any = None) -> None:
        self.policy = policy if isinstance(policy, LabelPolicy) else LabelPolicy()

    def adjudicate(self, hypotheses: Dict[str, Hypothesis], structured_alert: Any, investigation_history: List[Dict[str, Any]], decision: LabelDecision | None = None) -> Dict[str, Any]:
        del hypotheses, structured_alert, investigation_history
        if decision is None:
            now = datetime.now(timezone.utc)
            decision = self.policy.decide(
                CaseContext(
                    case_id="legacy-judge-without-evidence",
                    time_start=now,
                    time_end=now + timedelta(seconds=1),
                ),
                [],
            )
        return {
            "label": decision.label,
            "label_name": decision.label_name.value,
            "confidence": decision.confidence,
            "reasoning": decision.rationale,
            "attack_chain": [],
            "key_evidence": decision.supporting_evidence_ids,
            "uncertainties": [gap.description for gap in decision.information_gaps],
            "why_not_higher": decision.why_not_higher,
            "why_not_lower": decision.why_not_lower,
        }


if __name__ == "__main__":
    print("HypothesisManager 已加载；请通过 initialize_from_evidence 或 initialize_from_alert 调用。")
