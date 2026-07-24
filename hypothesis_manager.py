"""
假设管理引擎 (Hypothesis Management Engine)
职责：维护多个竞争假设，基于新证据进行贝叶斯更新，驱动研判结论或深度调查。

核心设计：
1. 多假设并行：同时维护"误报/扫描/成功入侵/内网横向"等竞争假设
2. 贝叶斯更新：每条新证据通过似然比(Likelihood Ratio)更新后验概率
3. 动态阈值：根据信息缺口自动触发"深度调查"或"结论生成"
4. 红队思维：自动生成反驳分析，模拟优秀分析师的自我质疑
"""

import math
import json
import uuid
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# 引入意图理解模块的输出
from alert_intent_parser import StructuredAlert, AlertEntity, EntityType

# ==================== 1. 核心数据结构 ====================
class HypothesisStatus(str, Enum):
    ACTIVE = "active"       # 活跃中，持续接收证据更新
    CONFIRMED = "confirmed" # 置信度超过阈值，已确认
    REJECTED = "rejected"   # 置信度低于阈值，已排除
    PENDING = "pending"     # 等待更多证据

class EvidenceType(str, Enum):
    SUPPORTING = "supporting"       # 支持该假设
    CONTRADICTING = "contradicting" # 反驳该假设
    NEUTRAL = "neutral"             # 中性/无关

@dataclass
class Evidence:
    """单条证据"""
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))[:8]
    source: str = ""                # 来源：intent_parser / investigation_agent / threat_intel
    raw_content: str = ""           # 证据原始内容
    related_entities: List[str] = field(default_factory=list)   # 关联的实体值
    evidence_type: EvidenceType = EvidenceType.NEUTRAL
    # 贝叶斯核心参数：似然比 LR = P(E|H) / P (E| ~H)
    # LR > 1: 支持假设; LR < 1 :反驳假设； LR = 1: 无关
    likelihood_ratio: float = 1.0

    weight: float = 1.0 # 证据权重（0-1），反映来源可靠性
    timestamp: datetime = field(default_factory=datetime.now)

    # 可解释性：为什么这条证据支持/反驳假设
    reasoning: str = ""

@dataclass
class Hypothesis:
    """单个假设"""
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""          # 假设名称，如"暴力破解成功"
    description: str = ""   # 假设名称，如"暴力破解成功"
    category: str = ""      # 大类：false_positive / reconnaissance / successful_attack / lateral_movement

    prior_probability: float = 0.5 # 先验概率，在没有任何证据之前，该假设为真的主观概率估计
    posterior_probability: float = 0.5 # 后验概率，表示根据已收集证据更新后，该假设为真的概率
    evidences: List[Evidence] = field(default_factory=list) # 用于保存假设的Eviden对象
    status: HypothesisStatus = HypothesisStatus.ACTIVE # 标记当前假设状态，默认ACTIVE表示改假设仍在接收证据并更新

    # 假设特有的“预期证据”：如果该假设为真，我们预期会发现什么
    except_evidence: List[str] = field(default_factory=list)
    #字符串列表，记录“该假设为真时理应存在但尚未发现的证据”。也用于对抗性分析，指出“缺失的关键证据”可能削弱假设可信度。
    missing_evidence: List[str] = field(default_factory=list)

    # 记录假设的创建时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 结论生成时的解释文本
    conclusion_reasoning: str = ""




@dataclass
class InvestigationRecommendation:
    """调查建议（输出给动态任务规划器）"""
    priority: Literal["critical","high","medium","low"]
    action: str = ""            #建议动作
    target_entities: List[str] = field(default_factory=list)
    rationle: str = ""          #为什么需要这个调查
    expected_outcome: str = ""  #预期能发现什么


# ==================== 2. 贝叶斯推理核心 ====================

class BayesianEngine:
    """
    贝叶斯推理引擎
    使用对数几率(Log-Odds)空间计算，避免多证据连乘下的数值下溢
    """
    @staticmethod
    def probability_to_log_odds(p: float) -> float:
        """概率 -> 对数几率"""
        p = max(0.0001, min(0.9999, p)) # 防止边界
        return math.log(p / (1 - p))

    @staticmethod
    def log_odds_to_probability(lo: float) -> float:
        """对数几率 -> 概率"""
        return 1.0 / (1.0 + math.exp(-lo))

    @staticmethod
    def likelihood_ration_to_log_weight(lr: float, weight: float = 1.0) -> float:
        """似然比 -> 对数权重"""
        # 限制LR范围防止极端值
        lr = max(0.01, min(100.0, lr))
        return weight * math.log(lr)

    def update(self, hypothesis: Hypothesis, evidence: Evidence) -> float:
        """
        对假设应用单条证据的贝叶斯更新
        返回更新后的后验概率
        """
        if hypothesis.status != HypothesisStatus.ACTIVE:
            return hypothesis.posterior_probability

        # 转换为对数几率攻坚
        prior_lo = self.probability_to_log_odds(hypothesis.posterior_probability)

        # 计算证据的对数权重
        log_weight = self.likelihood_ration_to_log_weight(
            evidence.likelihood_ratio,
            evidence.weight
        )

        # 更新：后验对数几率 = 先验对数几率 + 证据权重
        posterior_lo = prior_lo + log_weight

        # 转换回概率
        new_probability = self.log_odds_to_probability(posterior_lo)

        # 更新假设状态
        hypothesis.posterior_probability = new_probability
        hypothesis.evidences.append(evidence)
        hypothesis.updated_at = datetime.now()

        return new_probability

    def batch_update(self, hypothesis: Hypothesis, evidences: List[Evidence]) -> float:
        """批量更新"""
        for ev in evidences:
            self.update(hypothesis, ev)
        return hypothesis.posterior_probability

class LikelihoodRatioEstimator:
    """
    似然比评估器
    职责：根据证据内容和假设类型，评估 P(E|H) / P(E|~H)

    这是整个系统中，最需要“领域知识”的部分
    这里提供基于规则的基准实现，实际可扩展为LLM驱动或机器学习模型
    """

    # 预定义的证据模式 -> 各假设类型的似然比映射
    # 格式：{ “证据关键词/模式” ： { “假设类别”： LR} }
    EVIDENCE_PATTERNS = {
        # 误报相关指标
        "false_positive_indicators": {
            "false_positive": 5.0,
            "reconnaissance": 0.3,
            "successful_attack": 0.1,
            "lateral_movement": 0.1,
        },
        # 扫描/探测特征
        "scan_probe": {
            "false_positive": 0.5,
            "reconnaissance": 8.0,        # 扫描行为高度指向侦察
            "successful_attack": 0.4,
            "lateral_movement": 0.3,
        },
        # 成功利用/入侵指标
        "exploitation_success": {
            "false_positive": 0.1,
            "reconnaissance": 0.5,
            "successful_attack": 10.0,    # 成功利用强指向成功入侵
            "lateral_movement": 0.6,
        },
        # 横向移动指标
        "lateral_movement_sign": {
            "false_positive": 0.2,
            "reconnaissance": 0.4,
            "successful_attack": 0.8,
            "lateral_movement": 9.0,
        },
        # 恶意软件/植入物
        "malware_implant": {
            "false_positive": 0.1,
            "reconnaissance": 0.3,
            "successful_attack": 8.0,
            "lateral_movement": 2.0,
        },
        # 数据外传/窃取
        "data_exfiltration": {
            "false_positive": 0.2,
            "reconnaissance": 0.3,
            "successful_attack": 7.0,
            "lateral_movement": 1.5,
        },
    }

    def estimate(self, evidence_content: str, hypothesis_category: str) -> Tuple[float,str]:
        """
        评估证据对假设的似然比
        返回：(likelihood_ratio, reasoning)
        """

        content_lower = evidence_content.lower()

        # 规则匹配（可扩展为更复杂的NLP匹配）
        matched_pattern = None
        if any(k in content_lower for k in ["误报", "false positive", "baseline", "正常业务", "whitelist"]):
            matched_pattern = "false_poitive_indicators"
        elif any(k in content_lower for k in ["scan", "扫描", "probe", "探测", "port sweep"]):
            matched_pattern = "scan_probe"
        elif any(k in content_lower for k in ["exploit", "shell", "反弹", "reverse shell", "webshell", "getshell"]):
            matched_pattern = "exploitation_success"
        elif any(k in content_lower for k in ["lateral", "横向", "psexec", "wmiexec", "pass the hash", "ptt"]):
            matched_pattern = "lateral_movement_sign"
        elif any(k in content_lower for k in ["malware", "病毒", "trojan", "backdoor", "implant", "c2", "beacon"]):
            matched_pattern = "malware_implant"
        elif any(k in content_lower for k in ["exfil", "外传", "窃取", "download", "large transfer"]):
            matched_pattern = "data_exfiltration"

        if matched_pattern and matched_pattern in self.EVIDENCE_PATTERNS:
            lr_map = self.EVIDENCE_PATTERNS[matched_pattern]
            lr = lr_map.get(hypothesis_category, 1.0)
            reasoning = f"证据匹配模式 '{matched_pattern}', 对 '{hypothesis_category}'"
            return lr, reasoning

        # 默认中性证据
        return 1.0, "未匹配到已知模式，视为中性证据"

    def estimate_from_atomi_fact(self, fact: str, hypothesis_category: str) -> Tuple[float, str]:
        """直接基于原子事实评估"""
        return self.estimate(fact, hypothesis_category)

# ==================== 4. 假设管理器（主入口） ====================
class HypothesisManager:
    """
    假设管理器
    职责：维护假设空间，协调贝叶斯更新， 生成调查建议
    """

    # 研判结论阈值
    CONFIRMATION_THRESHOLD = 0.85  # 超过此值确认假设
    REJECTION_THRESHOLD = 0.15     # 低于此值排除假设
    DEEP_INVESTIGATION_THRESHOLD = 0.40 # 长期僵持在此区间触发深度调查

    def __init__(self):
        self.bayesian = BayesianEngine()
        self.lr_estimator = LikelihoodRatioEstimator()
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.investigation_history: List[Dict] = []

    # ---------- 4.1 初始化假设空间 ----------

    def initialize_from_alert(self, structured_alert: StructuredAlert) -> List[Hypothesis]:
        """
        基于意图理解模块的输出，初始化竞争假设空间
        这是假设引擎的入口点
        """
        self.hypotheses.clear()

        # 根据告警语义类别，生成相关の竞争假设
        semantics = structured_alert.semantics
        category = semantics.category

