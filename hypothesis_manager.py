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

        # 把“旧概率”变成“对数几率”   prior_lo = log( 旧概率 / (1 - 旧概率) )
        prior_lo = self.probability_to_log_odds(hypothesis.posterior_probability)

        # 第2步：把“证据倍数”变成“对数权重”
        log_weight = self.likelihood_ration_to_log_weight(
            evidence.likelihood_ratio,
            evidence.weight
        )

        # 第3步：对数空间里的加法（对应原始空间的乘法）
        posterior_lo = prior_lo + log_weight

        # 第4步：把结果变回“新概率”   new_probability = 1 / (1 + exp(-posterior_lo))
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
    EVIDENCE_PATTERNS = { # 证据倍数(似然比)查表
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

        # 通用假设：所有告警都考虑误报
        self._create_hypothesis(
            name="误报/正常业务行为",
            description="该告警由正常业务流量、配置错误或已知白名单行为触发",
            category="false_positive",
            prior=0.3, # 先验：30%的告警是误报（行业经验值）
            expected_evidence=["白名单匹配","业务高峰时段","已知正常IP"]
        )

        # 根据语义类别生成特定假设
        if category in ["credential_access", "intrusion"]:
            self._create_hypothesis(
                name="暴力破解/凭证攻击成功",
                description="攻击者成功通过暴力破解或凭证填充获取了有效凭据",
                category="successful_attack",
                prior=0.2,
                expected_evidence=["成功登录记录", "异常登录时间", "新地理位置"]
            )
            self._create_hypothesis(
                name="暴力破解尝试(未成功)",
                description="攻击者正在进行暴力破解，但尚未成功",
                category="reconnaissance",
                prior=0.25,
                expected_evidence=["大量失败登录","单一源IP","无成功记录"]
            )
        elif category in ["malware"]:
            self._create_hypothesis(
                name="主机已感染恶意软件",
                description="目标主机已执行恶意代码，可能已被控制",
                category="successful_attack",
                prior=0.2,
                expected_evidence=["可疑进程", "异常网络连接", "文件修改"]
            )
        
        elif category in ["reconnaissance"]:
            self._create_hypothesis(
                name="攻击者正在进行资产探测",
                description="攻击者正在扫描和探测目标网络，收集情报",
                category="reconnaissance",
                prior=0.35,
                expected_evidence=["端口扫描", "服务探测", "目录遍历"]
            )

        # 通用假设：成功入侵
        self._create_hypothesis(
            name="成功入侵并简历立足点",
            description="攻击者已成功突破防线，在目标环境建立了持久化访问",
            category="successful_attack",
            prior=0.15,
            expected_evidence=["后门/Webshell","C2通信","缺陷提升"]
        )

        # 通用假设：横向移动
        self._create_hypothesis(
            name="内网横向移动",
            description="攻击者已从初始立足点向内网其他主机扩散",
            category="lateral_movement",
            prior=0.1,
            expected_evidence=["内网连接异常", "凭证复用", "远程管理工具滥用"]
        )

        # 将意图理解模块的原子事实作为初始证据输入
        self._ingest_intent_facts(structured_alert)

        return list(self.hypotheses.values())

    def _create_hypothesis(self, name: str, description: str, category: str,
                            prior: float, excepted_evidence: List[str]=None):
        """辅助：创建单个假设"""
        h = Hypothesis(
            name=name,
            description=description,
            category=category,
            prior_probability=prior,
            posterior_probability=prior,
            except_evidence=excepted_evidence or []
        )
        self.hypotheses[h.hypothesis_id] = h
        return h

    def _ingest_intent_facts(self, structured_alert: StructuredAlert):
        """将意图理解模块的原子事实转换为证据，输入各假设"""
        for fact in structured_alert.atomic_facts:  #外循环：遍历从告警中取出的每一条原子事实
            # 为每个活跃假设评估这条证据
            for h in self.hypotheses.values():   # 内循环：遍历当前内存中所有的竞争假设
                # 将 (原子事实, 假设类别) 这个组合交给 LikelihoodRatioEstimator
                lr, reasoning = self.lr_estimator.estimate_from_atomic_fact(fact, h.category)

                ev = Evidence(  # 构建证据对象，将语义转化为数学参数
                    source="intent_parser",   # 打上来源标签，
                    raw_content=fact,         # 保留原始文本，供最终报告引用
                    evidence_type=EvidenceType.SUPPORTING if lr > 1.5 else(
                        EvidenceType.CONTRADICTING if lr < 0.7 else EvidenceType.NEUTRAL
                    ), # 这里引入缓冲区，避免微小的概率扰动过早地导致假设被排除或确认
                    likelihood_ratio=lr,
                    weight=0.8, # 意图解析器的产出是“原子事实”的二次加工产物，不如调查agent的一手日志可靠，所以设置权重为0.8                  reasoning=reasoning
                )
                self.bayesian.update(h, ev)  # 执行贝叶斯更新
        for gap in structured_alert.information_gaps:  # 遍历解析器识别出的“未知项”
            for h in self.hypotheses.values():         # 遍历所有假设
                h.missing_evidence.append(gap)

    # ---------- 4.2 证据输入接口 ----------

    def add_evidence(self, evidence: Evidence) -> Dict[str, float]:
        """
        外部Agent（调查Agent/取证Agent/情报Agent）调用此接口提交新证据
        返回各假设更新后的概率分布
        """
        results = {}

        for h in self.hypotheses.values():
            if h.status != HypothesisStatus.ACTIVE:
                continue

            # 评估证据对该假设的似然比
            lr, reasoning = self.lr_estimator.estimate(
                evidence.raw_content,
                h.category
            )
            evidence.likelihood_ratio = lr
            evidence.reasoning = reasoning

            # 更新
            new_p = self.bayesian.update(h, evidence)
            results[h.hypothesis_id] = new_p

            # 检查状态转换，每次更新后立即检查该假设的后验概率是否触及 0.85（确认）或 0.15（排除）阈值
            self._evaluate_hypothesis_status(h)

        self.investigation_history.append({  # 在完成对所有假设的更新后，将这次操作记录到历史列表中。
            "timestamp": datetime.now(),
            "evidence": evidence.raw_content[:100],
            "probabilities": {hid: self.hypotheses[hid].posterior_probability
                              for hid in self.hypotheses}
        })

        return results

    def _evaluate_hypothesis_status(self, h: Hypothesis):
        """评估假设是否需要改变状态"""
        if h.status != HypothesisStatus.ACTIVE:
            return

        p = h.posterior_probability

        if p >= self.CONFIRMATION_THRESHOLD:
            h.status = HypothesisStatus.CONFIRMED
            h.conclusion_reasoning = (
                f"假设 '{h.name}' 的后验概率达到 {p:.2%}, 超过确认阈值"
                f"{self.CONFIRMATION_THRESHOLD:.0%}，建议确认该假设。"
            )
        elif p <= self.REJECTION_THRESHOLD:
            h.status = HypothesisStatus.REJECTED
            h.conclusion_reasoning = (
                f"假设 '{h.name}' 的后验概率降至 {p:.2%}，低于排除阈值 "
                f"{self.REJECTION_THRESHOLD:.0%}，建议排除该假设。"
            )

    # ---------- 4.3 决策逻辑 ----------
    def get_status(self) -> Dict:
        """获取当前假设空间的整体状态"""
        active = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE]
        confirmed = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.CONFIRMED]
        rejected = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.REJECTED]

        # 检查是否形成明确结论
        has_conclusion = len(confirmed) > 0

        # 检查是否陷入僵持（深度调查触发条件）
        stalemate = False
        if not has_conclusion and len(active) >= 2: #至少还有 2 个假设在活跃竞争中
            probs = [h.posterior_probability for h in active]  # 提取所有活跃假设的后验概率值，存入列表。
            # 如果最高和次高概率接近，且都在中等区间
            if len(probs) >= 2:
                sorted_probs = sorted(probs, reverse=True) # 将概率从高到低排序。sorted_probs[0] 是最高概率，sorted_probs[1] 是次高概率
                if (sorted_probs[0] - sorted_probs[1] < 0.15 and  # 第一名和第二名的概率差距小于 15 个百分点
                    self.DEEP_INVESTIGATION_THRESHOLD < sorted_probs[0] < self.CONFIRMATION_THRESHOLD): #最高概率的值被夹在 0.40 和 0.85 之间。这排除了“虽胶着但双方都很弱”的情况，也排除了“虽胶着但头名已接近确认”的情况。
                    stalemate = True

            return {
                "has_conclusion": has_conclusion,            # 是否有结论
                "confirmed_hypotheses": confirmed,           # 已确认的假设对象列表
                "active_hypotheses": active,                 # 仍活跃的假设对象列表
                "rejected_hypotheses": rejected,             # 已排除的假设对象列表
                "stalemate": stalemate,                      # 是否陷入证据僵持
                "top_hypothesis": self.get_top_hypothesis(), # 当前概率最高的假设
                "entropy": self._calculate_entropy(),        # 信息熵值（不确定性度量）
            }

    def get_top_hypothesis(self) -> Optional[Hypothesis]:
        """获取当前概率最高的假设"""
        active = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE]
        if not active:
            return None
        return max(active, key=lambda h: h.posterior_probability) # Lambda 函数即匿名函数，以posterior_probability为key进行比较取出最大值

    def _calculate_entropy(self) -> float:
        """计算假设空间的信息熵（衡量不确定性）"""
        active = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE]
        if not active:
            return 0.0

        # 归一化概率
        total = sum(h.posterior_probability for h in active)
        entropy = 0.0
        for h in active:
            p = h.posterior_probability / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    # ---------- 4.4 调查建议生成（输出给动态任务规划器）----------

    def generate_investigation_recommendations(self) -> List[InvestigationRecommendation]:
        """
        基于当前假设状态，生成下一步调查建议
        这是假设引擎 -> 动态任务规划器 的关键接口
        """
        # 为什么几乎不用 elif: 确认攻击后，立即止损，停止一切调查性消耗,在没有最终结论时，系统倾向于饱和式任务下发。
        recommendations = []
        status = self.get_status() # 调用刚解析过的方法，获取当前假设空间的宏观快照
        #场景1： 已有假设确认 -> 生成处置建议
        if status["has_conclusion"]:
            for h in status["confirmed_hypotheses"]:
                recommendations.append(InvestigationRecommendation(
                    priority="critical",
                    action=f"确认假设 '{h.name}', 启动标准处置SOP",
                    rationle=h.conclusion_reasoning,
                    expected_outcome="完成攻击确认，进入响应阶段"
                ))
            return recommendations

        # 场景2：陷入僵持 -> 建议深度调查
        if status["stalemate"]:
            top_h = status["top_hypothesis"]
            recommendations.append(InvestigationRecommendation(
                priority="high",
                action="触发深度调查模式：扩大时间窗口和关联范围",
                rationale=f"假设 '{top_h.name}' (P={top_h.posterior_probability:.2%}) 与次优假设概率接近，"
                         f"信息熵={status['entropy']:.2f}，需要打破不确定性",
                expected_outcome="发现决定性证据，打破假设僵持"
            ))

        # 场景3：针对最高概率假设的缺失证据进行调查
        top_h = status["top_hypothesis"]
        if top_h and top_h.missing_evidence:
            for gap in top_h.missing_evidence[:2]:  # 遍历 missing_evidence 列表的前 2 项（防止一次性生成太多任务），生成精准的补查指令。
                recommendations.append(InvestigationRecommendation(
                    priority="high",
                    action=f"补充调查：{gap}", # 生成优先级为 high（高） 的“深度调查”指令。
                    # target_entities 试图从该假设最后一条证据的关联实体中提取 IP、用户名等，作为调查目标
                    target_entities=[e.value for e in top_h.evidences[-1].related_entities] if top_h.evidences else [],
                    rationle=f"假设 '{top_h.name}' 当前概率 {top_h.posterior_probability:.2%}" f"缺少 '{gap}' 将阻碍确认或排除",
                    expected_outcome=f"获取{gap}，显著改变假设概率"
                ))
        # 场景4： 针对高概率假设的验证（红队思维）
        active_sorted = sorted( # 先将所有活跃假设按概率从高到低排序。
            [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE],
            key=lambda h: h.posterior_probability,
            reverse=True
        )
        if len(active_sorted) >= 2:  # 活跃假设数量 ≥ 2。
            runner_up = active_sorted[1]
            recommendations.append(InvestigationRecommendation(
                priority="medium",  # 生成优先级为 medium（中） 的“红队验证”任务
                action=f"主动验证替代假设：{runner_up.name}",
                rationale=f"当前最优假设可能遗漏了支持 '{runner_up.name}' 的证据，"
                         f"主动寻找反驳证据以增强结论可信度",
                expected_outcome="排除或确认替代假设，降低研判不确定性"
            ))
        
        return recommendations # 返回最终建议清单

    # ---------- 4.5 红队思维：对抗性解释 ----------
    def generate_adversarial_analysis(self, target_hypothesis_id: Optional[str] = None) -> str:
        """
        生成对抗性解释：为什么这个假设可能是错的
        模拟“”红队思维
        """
        if target_hypothesis_id:
            h = self.hypotheses.get(target_hypothesis_id)
            if not h:
                return "假设目标不存在"
            else:
                # 默认对当前最优假设生成反驳
                h = self.get_top_hypothesis()
                if not h:
                    return "暂无活跃假设"

            # 收集反驳该假设的证据
            contradicting = [e for e in h.evidences if e.evidence_type == EvidenceType.CONTRADICTING]

            # 收集缺失的关键证据（如果假设为真应该存在但没找到）
            missing = h.missing_evidence

            lines = [
                f"## 对抗性分析：为什么「{h.name}」可能不正确",
                "",
                f"**当前概率**: {h.posterior_probability:.2%}",
                "",
                "### 1. 已发现的反驳证据"
            ]

            if contradicting:
                for e in contradicting:
                    lines.append(f"- {e.raw_content} (LR={e.likelihood_ratio:.2f})")
            else:
                lines.append("- 目前未发现直接反驳证据，但这本身不意味着假设正确")

            lines.extend([
                "",
                "### 2. 缺失的关键证据",
                "如果该假设为真，我们预期应该发现但尚未发现："
            ])
        
            
