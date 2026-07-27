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
import json, os
import uuid
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 引入意图理解模块的输出
from alert_intent_parser import StructuredAlert, AlertEntity, EntityType

load_dotenv()

"""大模型客户端初始化"""
llm = ChatOpenAI(
    model= os.getenv("LLM_MODEL_ID"),
    api_key= os.getenv("LLM_API_KEY"),
    base_url= os.getenv("LLM_BASE_URL"),
)


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
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
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
    expected_evidence: List[str] = field(default_factory=list)
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
    rationale: str = ""          #为什么需要这个调查
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

    def __init__(self, llm_client=None):
        self.llm = llm_client


    def estimate(self, evidence_content: str, hypothesis_category: str, 
                 hypothesis_name: str = "", hypothesis_desc: str = "") -> Tuple[float,str]:
        """
        评估证据对假设的似然比
        优先规则匹配，规则未命中时调用 LLM 做语义评估
        """

        content_lower = evidence_content.lower()

        # 第一层：规则匹配，保留性能优势（可扩展为更复杂的NLP匹配）
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

        # 第二层：LLM 语义评估（处理规则未覆盖的复杂证据）
        if self.llm:
            return self._llm_estimate(evidence_content, hypothesis_category, hypothesis_name, hypothesis_desc)

        # 默认中性证据
        return 1.0, "未匹配到已知模式，视为中性证据"
    

    def _llm_estimate(self, evidence_content: str, hypothesis_category: str,
                      hypothesis_name: str, hypothesis_desc: str) -> Tuple[float, str]:
        """调用 LLM 进行语义级似然比评估"""
        prompt = f"""你是一名安全分析专家，正在进行贝叶斯推理。

【假设信息】
- 假设类别: {hypothesis_category}
- 假设名称: {hypothesis_name}
- 假设描述: {hypothesis_desc}

【证据内容】
{evidence_content}

【任务】
请评估这条证据在"该假设为真" vs "该假设为假"两种情况下的出现概率之比（似然比 LR）。

输出要求（严格JSON）：
{{
    "likelihood_ratio": <float, 范围0.01-100.0>,
    "reasoning": "<简要解释为什么这条证据支持或反驳该假设>",
    "evidence_type": "supporting|contradicting|neutral"
}}

判断标准：
- LR > 3.0: 强支持该假设
- 1.0 < LR <= 3.0: 弱支持
- LR = 1.0: 中性无关
- 0.3 <= LR < 1.0: 弱反驳
- LR < 0.3: 强反驳

只输出JSON，不要任何解释。"""

        try:
            reponse = self.llm.chat(
                system_prompt = "你是一个严格的安全证据评估专家，只输出JSON。",
                user_prompt=prompt
            )
            #提取json
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            result = json.loads(response.strip())
            lr = float(result.get("likehood_ratio",1.0))
            lr = max(0.01, min(100.0,lr)) # 截断
            reasoning = result.get("reasoning", "LLM语义评估")
            return lr, reasoning
        except Exception as e:
            # LLM 失败时退回到中性
            return 1.0, f"LLM评估失败({str(e)})，回退为中性证据"

    def estimate_from_atomic_fact(self, fact: str, hypothesis_category: str,
                                   hypothesis_name: str = "", hypothesis_desc: str = "") -> Tuple[float, str]:
        """直接基于原子事实评估"""
        return self.estimate(fact, hypothesis_category, hypothesis_name, hypothesis_desc)

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

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.bayesian = BayesianEngine()
        # 传入 LLM 给似然比评估器
        self.lr_estimator = LikelihoodRatioEstimator(llm_client=llm_client)
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
                            prior: float, expected_evidence: List[str]=None):
        """辅助：创建单个假设"""
        h = Hypothesis(
            name=name,
            description=description,
            category=category,
            prior_probability=prior,
            posterior_probability=prior,
            expected_evidence=expected_evidence or []
        )
        self.hypotheses[h.hypothesis_id] = h
        return h

    def _ingest_intent_facts(self, structured_alert: StructuredAlert):
        """将意图理解模块的原子事实转换为证据，输入各假设"""
        for fact in structured_alert.atomic_facts:  #外循环：遍历从告警中取出的每一条原子事实
            # 为每个活跃假设评估这条证据
            for h in self.hypotheses.values():   # 内循环：遍历当前内存中所有的竞争假设
                # 将 (原子事实, 假设类别) 这个组合交给 LikelihoodRatioEstimator
                # ===== 修改：传入假设名称和描述，让 LLM 理解上下文 =====
                lr, reasoning = self.lr_estimator.estimate_from_atomic_fact(
                    fact, 
                    h.category,
                    hypothesis_name=h.name,
                    hypothesis_desc=h.description
                )

                ev = Evidence(  # 构建证据对象，将语义转化为数学参数
                    source="intent_parser",   # 打上来源标签，
                    raw_content=fact,         # 保留原始文本，供最终报告引用
                    evidence_type=EvidenceType.SUPPORTING if lr > 1.5 else(
                        EvidenceType.CONTRADICTING if lr < 0.7 else EvidenceType.NEUTRAL
                    ), # 这里引入缓冲区，避免微小的概率扰动过早地导致假设被排除或确认
                    likelihood_ratio=lr,
                    weight=0.8, # 意图解析器的产出是“原子事实”的二次加工产物，不如调查agent的一手日志可靠，所以设置权重为0.8                  
                    reasoning=reasoning
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
        # print(f"[DEBUG] self type: {type(self)}")
        # print(f"[DEBUG] self.get_status type: {type(self.get_status)}")
        # print(f"[DEBUG] self.get_status is method? {hasattr(self.get_status, '__call__')}")
        # 为什么几乎不用 elif: 确认攻击后，立即止损，停止一切调查性消耗,在没有最终结论时，系统倾向于饱和式任务下发。
        recommendations = []
        status = self.get_status() # 调用刚解析过的方法，获取当前假设空间的宏观快照
        # print(f"[DEBUG] status = {status}")
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

        # ===== LLM 增强：动态认知规划 =====
        if self.llm and status["active_hypotheses"]:
            try:
                llm_recs = self._llm_generate_recommendations(status)
                if llm_recs:
                    recommendations.extend(llm_recs)
                    return recommendations  # LLM 生成成功则直接返回
            except Exception as e:
                # LLM 失败时降级到规则逻辑
                pass

        # ===== 降级：规则模板逻辑 =====
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
                # ---------- 兼容性处理 target_entities ----------
                if top_h.evidences:
                    related = top_h.evidences[-1].related_entities
                    target_entities = [
                        e.value if hasattr(e, 'value') else e
                        for e in related
                    ]
                else:
                    target_entities = []
                # ------------------------------------------------
                recommendations.append(InvestigationRecommendation(
                    priority="high",
                    action=f"补充调查：{gap}", # 生成优先级为 high（高） 的“深度调查”指令。
                    # target_entities 试图从该假设最后一条证据的关联实体中提取 IP、用户名等，作为调查目标     
                    target_entities=target_entities,  # 使用处理后的列表
                    rationale=f"假设 '{top_h.name}' 当前概率 {top_h.posterior_probability:.2%}" f"缺少 '{gap}' 将阻碍确认或排除",
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

    def _llm_generate_recommendations(self, status:Dict) -> List[InvestigationRecommendation]:
        """调用 LLM 生成动态调查建议"""
        # 构建假设状态摘要
        hypo_summary = []
        for h in sorted(status["active_hypotheses"], 
                       key=lambda x: x.posterior_probability, reverse=True)[:4]:
            gaps_text = ", ".join(h.missing_evidence[:3]) if h.missing_evidence else "无"
            hypo_summary.append(
                f"假设: {h.name}(P={h.posterior_probability:.2%}, 类别:{h.category})\n"
                f"  描述: {h.description}\n"
                f"  缺失证据: {gaps_text}\n"
                f"  已有证据数: {len(h.evidences)}"
            )

        prompt = f"""你是一名资深安全调查指挥官，当前多个竞争假设处于活跃状态，请制定最优调查策略。

【当前假设空间状态】
信息熵: {status['entropy']:.3f} (越高表示不确定性越大)
是否僵持: {'是' if status['stalemate'] else '否'}

【活跃假设详情】
{chr(10).join(hypo_summary)}

【任务】
请输出下一步的 1-3 条调查建议。每条建议必须包含：
1. priority: critical/high/medium/low
2. action: 具体可执行的动作描述
3. rationale: 为什么现在要做这个调查（关联到假设概率和证据缺口）
4. expected_outcome: 预期能获取什么证据，以及该证据如何影响假设概率

输出格式（严格JSON数组）：
[
  {{
    "priority": "high",
    "action": "具体动作",
    "rationale": "理由",
    "expected_outcome": "预期结果"
  }}
]

注意：
- 优先调查能"打破假设僵持"或"填补最大信息缺口"的动作
- 如果某个假设已接近确认阈值(>0.7)，建议优先验证其关键缺失证据
- 避免建议已经做过或明显无效的调查
- 只输出JSON数组，不要其他内容"""
        response = self.llm.chat(
            system_prompt="你是一个安全调查策略规划专家，只输出JSON。",
            user_prompt=prompt
        )
        
        # 解析JSON
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        recs_data = json.loads(response.strip())
        results = []
        for rec in recs_data:
            results.append(InvestigationRecommendation(
                priority=rec.get("priority", "medium"),
                action=rec.get("action", ""),
                rationale=rec.get("rationale", ""),
                expected_outcome=rec.get("expected_outcome", "")
            ))
        return results




    # ---------- 4.5 红队思维：对抗性解释 ----------
    def generate_adversarial_analysis(self, target_hypothesis_id: Optional[str] = None) -> str:
        """
        生成对抗性解释：为什么这个假设可能是错的
        模拟“”红队思维
        """
        if target_hypothesis_id:
            h = self.hypotheses.get(target_hypothesis_id) # 选出可能性最高的一个赋值给h
            if not h:
                return "假设目标不存在"
        else:
            # 默认对当前最优假设生成反驳
            h = self.get_top_hypothesis()
            if not h:
                return "暂无活跃假设"

        # 收集反驳该假设的证据，从目标假设 h 已记录的所有证据 evidences 中，筛选出类型为 CONTRADICTING（反驳）的条目，生成列表 contradicting。
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
            for e in contradicting: # 如果存在反驳证据，逐条添加到line列表中，格式为证据原始内容+该条证据对该假设的似然比
                lines.append(f"- {e.raw_content} (LR={e.likelihood_ratio:.2f})")
        else:
            lines.append("- 目前未发现直接反驳证据，但这本身不意味着假设正确")

        lines.extend([
            "",
            "### 2. 缺失的关键证据",
            "如果该假设为真，我们预期应该发现但尚未发现："
        ])
        if missing:
            for m in missing:
                lines.append(f"- {m}")
        else:
            lines.append("- 暂无明确缺失证据记录")

        lines.extend([
        "",
        "### 3. 替代解释",
        "以下替代假设同样可能解释当前证据："
        ])
        alternatives = [
            ah for ah in self.hypotheses.values()
            # 从全局假设字典中提取代替假设：排除当前正在分析的目标假设，排除已标记为REJECTED的假设（因为已确认不成立）
            if ah.hypothesis_id != h.hypothesis_id and ah.status != HypothesisStatus.REJECTED
        ]

        alternatives.sort(key=lambda x: x.posterior_probability, reverse=True) # 按后验概率从高到低对替代假设排序，突出最可能混淆判断的假设。

        for alt in alternatives[:3]: #选取前三名替代假设，输出名称、当前概率及描述。
            lines.append(f"- **{alt.name}** (P={alt.posterior_probability:.2%}): {alt.description}")

        lines.extend([
            "",
            "### 4. 建议",
            "在确认该假设前，建议优先补充上述缺失证据，或找到能显著区分该假设与替代假设的决定性证据。"
        ])

        return "\n".join(lines)

        # ---------- 4.6 最终报告生成 ----------
    
    def generate_report(self) -> Dict:
        """生成当前研判状态的完整报告"""
        status = self.get_status()

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_hypotheses": len(self.hypotheses),
                "active": len(status["active_hypotheses"]),
                "confirmed": len(status["confirmed_hypotheses"]),
                "rejected": len(status["rejected_hypotheses"]),
                "uncertainty_entropy": round(status["entropy"], 4),
            },
            "hypotheses": [],
            "top_recommendations": [],
            "adversarial_analysis": "",
            "investigation_history": self.investigation_history[-5:]  # 最近5步
        }

        # 所有假设详情
        all_h = sorted(self.hypotheses.values(), 
                      key=lambda h: h.posterior_probability, 
                      reverse=True)
        for h in all_h:
            report["hypotheses"].append({
                "id": h.hypothesis_id,
                "name": h.name,
                "category": h.category,
                "status": h.status.value,
                "prior": round(h.prior_probability, 4),
                "posterior": round(h.posterior_probability, 4),
                "evidence_count": len(h.evidences),
                "conclusion": h.conclusion_reasoning if h.status != HypothesisStatus.ACTIVE else None
            })
        
        # 调查建议
        recs = self.generate_investigation_recommendations()
        report["top_recommendations"] = [
            {
                "priority": r.priority,
                "action": r.action,
                "rationale": r.rationale,
                "expected_outcome": r.expected_outcome
            }
            for r in recs
        ]
        
        # 对抗性分析（对最优假设）
        report["adversarial_analysis"] = self.generate_adversarial_analysis()
        
        return report

# ==================== 5. 使用示例 ====================

if __name__ == "__main__":
    # 模拟意图理解模块的输出
    from alert_intent_parser import AlertSemantics, ChatOpenAIAdapter

    # 初始化 LLM 适配器
    adapter = ChatOpenAIAdapter(llm)
    
    # 传入 LLM 客户端
    hm = HypothesisManager(llm_client=adapter)
    
    test_alert = StructuredAlert(
        alert_id="ALERT-TEST-LLM-001",
        raw_alert="[HIGH] 异常行为告警：用户 admin 在凌晨3点通过VPN从非常用地理位置登录，随后访问了敏感财务目录",
        source_system="SIEM",
        semantics=AlertSemantics(
            category="credential_access",
            tactic="异常登录行为",
            severity="high",
            intent_tags=["suspicious_login", "off_hours_access"]
        ),
        entities=[],
        atomic_facts=[
            # 规则能匹配的（测试规则路径是否正常）
            "源IP 45.33.22.11 来自外部网络",
            # 规则无法匹配的（强制触发 LLM 评估）
            "用户 admin 在凌晨3点通过VPN从非常用地理位置登录",
            "登录后访问了敏感财务目录，访问模式与历史基线偏离87%",
            "会话持续期间未触发DLP告警，但下载了3个加密压缩包",
            # 更模糊的描述，测试 LLM 语义理解
            "该IP在过去72小时内没有威胁情报记录，但ASN归属地为高风险地区"
        ],
        information_gaps=[
            "缺少该VPN会话的完整命令执行记录",
            "缺少敏感文件的下载后去向",
            "缺少admin用户的历史正常登录基线"
        ]
    )
    
    print("=" * 70)
    print("【测试1】初始化假设空间 + LLM 语义评估原子事实")
    print("=" * 70)
    
    hypotheses = hm.initialize_from_alert(test_alert)
    
    print("\n各假设后验概率（LLM应已参与评估模糊证据）：")
    for h in sorted(hypotheses, key=lambda x: x.posterior_probability, reverse=True):
        print(f"\n  [{h.hypothesis_id}] {h.name}")
        print(f"    类别: {h.category}")
        print(f"    后验概率: {h.posterior_probability:.2%}")
        print(f"    状态: {h.status.value}")
        # 打印证据详情，查看是否有 LLM 生成的 reasoning
        llm_evidence = [e for e in h.evidences if "LLM" in e.reasoning or "语义" in e.reasoning]
        if llm_evidence:
            print(f"    🤖 LLM评估证据数: {len(llm_evidence)}")
            for e in llm_evidence[:2]:
                print(f"       - {e.raw_content[:40]}... | LR={e.likelihood_ratio:.2f} | {e.reasoning[:50]}")

    # ========== 步骤4：提交规则无法匹配的外部证据，测试 LLM 评估 ==========
    print("\n" + "=" * 70)
    print("【测试2】外部Agent提交模糊证据（强制走LLM评估）")
    print("=" * 70)
    
    # 这条证据不含任何规则关键词，LLM 必须自己判断语义
    vague_evidence = Evidence(
        source="investigation_agent",
        raw_content="在admin用户的家目录下发现名为 '.config_backup_2024' 的隐藏目录，"
                   "其中包含大量与admin日常工作无关的源代码文件，"
                   "目录创建时间恰好与异常VPN会话时间吻合，"
                   "但文件内容经初步检查未发现明显的恶意特征码",
        related_entities=["admin"],
        weight=1.0
    )
    
    results = hm.add_evidence(vague_evidence)
    print(f"\n提交证据: {vague_evidence.raw_content[:60]}...")
    print("更新后概率分布:")
    for hid, p in results.items():
        h = hm.hypotheses[hid]
        print(f"  [{h.name}] {p:.2%} | 状态: {h.status.value}")

    # ========== 步骤5：测试 LLM 动态调查建议 ==========
    print("\n" + "=" * 70)
    print("【测试3】LLM 动态生成调查建议")
    print("=" * 70)
    
    recs = hm.generate_investigation_recommendations()
    if recs:
        for i, r in enumerate(recs, 1):
            print(f"\n建议 {i} [优先级: {r.priority}]")
            print(f"  动作: {r.action}")
            print(f"  理由: {r.rationale}")
            print(f"  预期: {r.expected_outcome}")
            # 如果建议是LLM生成的，通常会更具体、有针对性
            if len(r.action) > 30 and "调查" in r.action:
                print("  ✅ 疑似LLM生成（内容较长且语义丰富）")
    else:
        print("⚠️ 未生成建议，可能LLM调用失败或已确认假设")

    # ========== 步骤6：测试 LLM 对抗性分析（红队思维） ==========
    print("\n" + "=" * 70)
    print("【测试4】LLM 对抗性分析（红队思维）")
    print("=" * 70)
    
    adversarial_report = hm.generate_adversarial_analysis()
    print(adversarial_report)
    
    # 简单判断：如果报告包含"认知偏差"、"证据链"、"替代场景"等词，说明LLM生效
    if any(kw in adversarial_report for kw in ["认知偏差", "证据链", "替代场景", "决定性检验"]):
        print("\n✅ LLM 对抗性分析已生效（检测到LLM特有表述）")
    elif "假设目标不存在" in adversarial_report or "暂无活跃假设" in adversarial_report:
        print("\n⚠️ 对抗性分析未触发LLM，可能假设已被确认/排除")
    else:
        print("\n⚠️ 对抗性分析可能未调用LLM（回退到模板输出）")

    # ========== 步骤7：生成完整报告 ==========
    print("\n" + "=" * 70)
    print("【测试5】完整研判报告")
    print("=" * 70)
    report = hm.generate_report()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    
    # ========== 最终判断 ==========
    print("\n" + "=" * 70)
    print("【LLM 工作状态判定】")
    print("=" * 70)
    
    # 检查是否有任何证据的reasoning包含LLM相关字样
    all_evidences = []
    for h in hm.hypotheses.values():
        all_evidences.extend(h.evidences)
    
    llm_reasonings = [e for e in all_evidences if e.reasoning and ("LLM" in e.reasoning or "语义" in e.reasoning)]
    
    if llm_reasonings:
        print(f"✅ LLM 已参与推理（共 {len(llm_reasonings)} 条证据经过LLM评估）")
    else:
        print("❌ LLM 未参与推理，所有证据均通过规则匹配处理")
        print("   可能原因：")
        print("   1. 证据文本被规则关键词匹配到了（走规则分支）")
        print("   2. HypothesisManager 初始化时未传入 llm_client")
        print("   3. LLM API 调用失败（检查网络/密钥）")


        
            
