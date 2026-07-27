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
import json, os, sys, io
import uuid
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from alert_intent_parser import AlertSemantics, ChatOpenAIAdapter
os.environ['PYTHONUTF8'] = '1'

# 引入意图理解模块的输出
from alert_intent_parser import StructuredAlert, IntentUnderstandingEngine, EntityType

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

class HypothesisCategory(str, Enum):
    FALSE_POSITIVE = "false_positive"           # 标签1
    SUSPECTED_ATTACK = "suspected_attack"       # 标签2
    ATTACK_BLOCKED = "attack_blocked"           # 标签3
    ATTACK_SUCCEEDED_NOT_COMPROMISED = "attack_succeeded_not_compromised"  # 标签4
    COMPROMISED = "compromised"                 # 标签5

# 标签到假设类别的映射
LABEL_TO_CATEGORY = {
    1: HypothesisCategory.FALSE_POSITIVE,
    2: HypothesisCategory.SUSPECTED_ATTACK,
    3: HypothesisCategory.ATTACK_BLOCKED,
    4: HypothesisCategory.ATTACK_SUCCEEDED_NOT_COMPROMISED,
    5: HypothesisCategory.COMPROMISED,
}

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
    似然比评估器 —— 重构为 LLM主路径，规则预筛辅助
    """
    def __init__(self, llm_client=None):
        self.llm = llm_client
        # 规则模式保留，但仅用于快速预分类，不直接输出LR
        self.QUICK_PATTERNS = {
            "false_positive_indicators": ["误报", "false positive", "baseline", "正常业务", "whitelist"],
            "scan_probe": ["scan", "扫描", "probe", "探测", "port sweep"],
            "exploitation_success": ["exploit", "shell", "反弹", "reverse shell", "webshell", "getshell"],
            "lateral_movement_sign": ["lateral", "横向", "psexec", "wmiexec", "pass the hash"],
            "malware_implant": ["malware", "病毒", "trojan", "backdoor", "implant", "c2", "beacon"],
            "data_exfiltration": ["exfil", "外传", "窃取", "download", "large transfer"],
        }

    def estimate(self, evidence_content: str, hypothesis_category: str, 
                 hypothesis_name: str = "", hypothesis_desc: str = "") -> Tuple[float, str]:
        """
        LLM主路径：所有证据都经过LLM语义评估
        规则仅用于给LLM提供上下文线索，不直接决定LR
        """
        # Step 1: 规则快速预分类（给LLM的prompt提供线索，不输出结果）
        matched_tags = []
        content_lower = evidence_content.lower()
        for tag, keywords in self.QUICK_PATTERNS.items():
            if any(k in content_lower for k in keywords):
                matched_tags.append(tag)
        
        # Step 2: 强制走LLM评估（核心修改！）
        if self.llm:
            return self._llm_estimate(
                evidence_content, 
                hypothesis_category, 
                hypothesis_name, 
                hypothesis_desc,
                matched_tags=matched_tags
            )
        
        # 仅在LLM完全不可用时降级到规则（比赛时应避免走到这里）
        return self._rule_fallback(evidence_content, hypothesis_category, matched_tags)

    def _llm_estimate(self, evidence_content: str, hypothesis_category: str,
                      hypothesis_name: str, hypothesis_desc: str,
                      matched_tags: List[str] = None) -> Tuple[float, str]:
        """LLM语义评估 —— 这是核心推理能力"""
        
        tags_hint = f"规则预分类标签: {matched_tags}" if matched_tags else "无规则预分类"
        
        prompt = f"""你是一名资深安全分析专家，正在进行贝叶斯推理研判。

【假设信息】
- 假设类别: {hypothesis_category}
- 假设名称: {hypothesis_name}
- 假设描述: {hypothesis_desc}

【证据内容】
{evidence_content}

【辅助线索】
{tags_hint}

【关键任务】
请深入分析这条证据在"该假设为真" vs "该假设为假"两种情况下的出现概率之比（似然比 LR）。

分析要求：
1. 不要仅看关键词，要理解证据的深层语义
2. 对于HTTP响应，要分析状态码、响应体内容、重定向目标的真实含义
3. 对于攻击payload，要判断是"尝试"还是"成功执行"
4. 考虑攻击者的TTPs（战术、技术、程序）

输出格式（严格JSON）：
{{
    "likelihood_ratio": <float, 范围0.01-100.0>,
    "reasoning": "<详细解释：为什么这条证据支持或反驳该假设，至少50字>",
    "evidence_type": "supporting|contradicting|neutral",
    "attack_stage": "<如: reconnaissance/weaponization/delivery/exploitation/installation/c2/actions_on_objective/unknown>",
    "confidence": <float, 0-1>
}}

判断标准：
- LR > 10: 极强支持
- 3.0 < LR <= 10: 强支持  
- 1.5 < LR <= 3.0: 中等支持
- 0.7 <= LR <= 1.5: 中性/无关
- 0.3 <= LR < 0.7: 中等反驳
- 0.1 <= LR < 0.3: 强反驳
- LR < 0.1: 极强反驳

只输出JSON，不要任何解释。"""

        try:
            # ===== 修复Bug: 统一变量名 =====
            response = self.llm.chat(
                system_prompt="你是一个严格的安全证据评估专家，只输出JSON。",
                user_prompt=prompt
            )
            
            # 提取json
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            result = json.loads(response.strip())
            
            # ===== 修复Bug: 正确的字段名 =====
            lr = float(result.get("likelihood_ratio", 1.0))
            lr = max(0.01, min(100.0, lr))
            reasoning = result.get("reasoning", "LLM语义评估")
            
            # 如果LLM给出了高质量的攻击阶段分析，附加到reasoning
            attack_stage = result.get("attack_stage", "")
            if attack_stage and attack_stage != "unknown":
                reasoning += f" [攻击阶段: {attack_stage}]"
            
            return lr, reasoning
            
        except Exception as e:
            # LLM失败时降级到规则，但记录错误
            print(f"[WARN] LLM评估失败: {e}，降级到规则匹配")
            return self._rule_fallback(evidence_content, hypothesis_category, matched_tags)

    def _rule_fallback(self, evidence_content, hypothesis_category, matched_tags):
        """规则降级方案 —— 仅在LLM完全不可用时使用"""
        # ... 原有规则逻辑 ...
        lr_map = {
            "false_positive": 5.0, "reconnaissance": 0.3,
            "successful_attack": 0.1, "lateral_movement": 0.1
        }
        lr = lr_map.get(hypothesis_category, 1.0)
        return lr, f"规则降级评估 (LLM不可用) | 预分类: {matched_tags}"



class LLMResponseAnalyzer:
    """
    LLM驱动的HTTP响应语义分析器
    职责：判断一次HTTP攻击交互的真实结果（成功/拦截/失败/无法判断）
    这是规则无法做到的——比如302重定向可能是WAF拦截，也可能是正常跳转
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def analyze(self, request_headers: str, request_body: str, 
                response_headers: str, response_body: str,
                alert_name: str) -> Dict:
        """
        让LLM分析一次完整的HTTP交互，判断攻击结果
        """
        prompt = f"""你是一名Web安全渗透测试专家。请分析以下HTTP交互，判断攻击的真实结果。

【告警类型】
{alert_name}

【请求头】
{request_headers[:800]}

【请求体】
{request_body[:800] if request_body else "无"}

【响应头】
{response_headers[:800]}

【响应体】
{response_body[:800] if response_body else "无"}

【分析任务】
1. 攻击payload分析：这是什么类型的攻击？payload的意图是什么？
2. 响应语义分析：服务器响应的真实含义是什么？
   - 302重定向到anonym.jsp：是WAF拦截还是正常跳转？
   - 400错误：是请求被拦截还是目标不存在？
   - 500错误：是攻击触发了异常还是服务器正常报错？
   - 200 OK：响应体是否包含攻击成功的证据（如/etc/passwd内容、whoami结果）？
3. 攻击结果判断：
   - blocked: 被WAF/IPS/RASP拦截，攻击未到达应用层
   - failed: 攻击到达应用层但执行失败（如SQL语法错误、路径不存在）
   - suspicious: 无法确定，响应模糊（如500错误可能是成功也可能是失败）
   - success: 攻击成功执行（如命令回显、敏感数据泄露、文件上传成功）

输出严格JSON：
{{
    "attack_type": "<攻击类型>",
    "payload_intent": "<payload意图>",
    "response_meaning": "<响应真实含义>",
    "result": "blocked|failed|suspicious|success",
    "confidence": <0-1>,
    "indicators": ["指标1", "指标2"],
    "reasoning": "<详细推理过程，至少80字>"
}}"""

        try:
            response = self.llm.chat(
                system_prompt="你是Web安全专家，只输出JSON。",
                user_prompt=prompt
            )
            
            # 提取JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            # 降级到规则分析
            return self._rule_analyze(response_headers, response_body)
    
    def _rule_analyze(self, response_headers, response_body):
        """规则降级"""
        status_code = 0
        if response_headers:
            match = __import__('re').search(r'HTTP/\d\.\d\s+(\d+)', response_headers)
            if match:
                status_code = int(match.group(1))
        
        if status_code in [403, 404, 400]:
            return {"result": "blocked", "confidence": 0.6, "reasoning": "基于状态码规则判断"}
        elif status_code == 302:
            return {"result": "blocked", "confidence": 0.5, "reasoning": "302重定向，可能是拦截"}
        elif status_code == 200:
            return {"result": "suspicious", "confidence": 0.5, "reasoning": "200响应需进一步分析"}
        else:
            return {"result": "suspicious", "confidence": 0.3, "reasoning": "证据不足"}


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
        # ===== 修复Bug: 只初始化一次，保留llm_client =====
        self.lr_estimator = LikelihoodRatioEstimator(llm_client=llm_client)
        # 删除这行: self.lr_estimator = LikelihoodRatioEstimator()
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.investigation_history: List[Dict] = []
        # 新增：LLM裁决引擎
        self.judge_engine = LLMJudgeEngine(llm_client) if llm_client else None

    # ---------- 4.1 初始化假设空间 ----------

    def initialize_from_alert(self, structured_alert: StructuredAlert) -> List[Hypothesis]:
        """
        基于意图理解模块的输出，初始化竞争假设空间
        这是假设引擎的入口点
        """
        self.hypotheses.clear()

        # 分析原子事实，提取关键指标
        facts_text = " ".join(structured_alert.atomic_facts)
        has_attempt_only = "attack_state: attempt" in facts_text and "attack_state: success" not in facts_text
        has_4xx_5xx = any(code in facts_text for code in ["400", "403", "404", "500"])
        has_2xx = any(code in facts_text for code in ["200 OK", "201"])
        has_cmd_payload = any(k in facts_text for k in ["curl", "whoami", "cat ", "/etc/passwd", "ping "])
        has_webshell_indicator = any(k in facts_text for k in ["jsp?", "php?", "asp?", "shell"])
        
        # 假设1: 误报（标签1）
        self._create_hypothesis(
            name="误报/正常业务扫描",
            description="告警由安全扫描器、渗透测试或正常业务触发，无实际危害",
            category=HypothesisCategory.FALSE_POSITIVE,
            prior=0.15,
            expected_evidence=["WAF拦截日志", "已知扫描器IP白名单", "业务系统正常响应"]
        )
        
        # 假设2: 疑似攻击（标签2）
        self._create_hypothesis(
            name="疑似遭受攻击（未确认影响）",
            description="存在攻击特征，但缺乏攻击成功或拦截的明确证据",
            category=HypothesisCategory.SUSPECTED_ATTACK,
            prior=0.20,
            expected_evidence=["攻击payload", "异常请求模式", "无明确成功/拦截证据"]
        )
        
        # 假设3: 攻击被拦截（标签3）—— 针对 example.json 的最可能假设
        self._create_hypothesis(
            name="攻击被WAF/IPS拦截",
            description="攻击尝试已被安全设备识别并阻断，未到达应用层",
            category=HypothesisCategory.ATTACK_BLOCKED,
            prior=0.35 if (has_attempt_only and has_4xx_5xx) else 0.20,
            expected_evidence=["302重定向到拦截页面", "400/403/404响应", "WAF拦截记录", "无成功回显"]
        )
        
        # 假设4: 攻击成功但未失陷（标签4）
        self._create_hypothesis(
            name="攻击成功但未建立持久化",
            description="攻击payload已执行或文件已落地，但未发现持久化/C2/横向移动",
            category=HypothesisCategory.ATTACK_SUCCEEDED_NOT_COMPROMISED,
            prior=0.15 if has_2xx else 0.10,
            expected_evidence=["200 OK响应含敏感数据", "命令执行回显", "文件上传成功", "无WebShell/C2证据"]
        )
        
        # 假设5: 已失陷（标签5）
        self._create_hypothesis(
            name="主机已失陷",
            description="已确认命令执行、持久化、C2通信或横向移动",
            category=HypothesisCategory.COMPROMISED,
            prior=0.10 if (has_cmd_payload and has_2xx) else 0.05,
            expected_evidence=["WebShell访问记录", "C2外联连接", "异常进程启动", "凭证访问", "横向移动痕迹"]
        )
        
        # 注入原子事实作为初始证据
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

class LLMJudgeEngine:
    """
    LLM驱动的最终标签裁决引擎
    输入：完整证据链 + 假设概率分布
    输出：1-5标签 + 详细推理
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def adjudicate(self, hypotheses: Dict[str, Hypothesis], 
                   structured_alert: StructuredAlert,
                   investigation_history: List[Dict]) -> Dict:
        """
        让LLM做最终标签裁决 —— 这是最关键的认知决策
        """
        # 构建证据摘要
        evidence_summary = []
        for h in sorted(hypotheses.values(), key=lambda x: x.posterior_probability, reverse=True):
            ev_list = [f"  - {e.raw_content[:80]} (LR={e.likelihood_ratio:.2f})" 
                      for e in h.evidences[-5:]]  # 最近5条
            evidence_summary.append(
                f"假设: {h.name} (P={h.posterior_probability:.2%})\n"
                f"类别: {h.category}\n"
                f"关键证据:\n" + "\n".join(ev_list) + "\n"
                f"缺失证据: {', '.join(h.missing_evidence[:3]) if h.missing_evidence else '无'}"
            )
        
        # 构建原子事实摘要
        fact_summary = "\n".join([f"- {f}" for f in structured_alert.atomic_facts[:15]])
        
        prompt = f"""你是一名首席安全分析师，正在进行最终告警研判裁决。

【赛题标签定义】
1. false_positive(误报): 行为可由正常业务解释，无可信攻击证据
2. suspected_attack(疑似): 存在攻击特征，但不足以确认攻击已执行或产生影响
3. attack_blocked(被拦截): 已确认攻击尝试，但在执行/落地/达到目标前被拦截
4. attack_succeeded_not_compromised(成功未失陷): 载荷已投递/落地，但没有证据证明被执行、建立持久化或取得控制
5. compromised(已失陷): 已确认命令执行、持久化、凭据访问、C2、隧道通信、横向移动、数据访问

【当前假设空间状态】
{chr(10).join(evidence_summary)}

【关键原子事实】
{fact_summary}

【信息缺口】
{chr(10).join(['- ' + g for g in structured_alert.information_gaps])}

【裁决任务】
1. 基于上述证据，选择最符合的label（1-5）
2. 必须遵循保守原则：如无法区分相邻标签，采用较低标签
3. 在uncertainties中说明缺失的证据
4. 还原攻击链（按时间顺序列出关键步骤）

输出严格JSON：
{{
    "label": <1-5的整数>,
    "label_name": "<false_positive|suspected_attack|attack_blocked|attack_succeeded_not_compromised|compromised>",
    "confidence": <0-1>,
    "reasoning": "<详细推理过程，至少150字，说明为什么选这个标签而不是相邻标签>",
    "attack_chain": [
        {{"step": 1, "time": "<时间>", "action": "<攻击动作>", "evidence": "<证据>", "result": "<成功/失败/拦截>"}}
    ],
    "key_evidence": ["证据1", "证据2"],
    "uncertainties": ["缺失的证据1", "缺失的证据2"],
    "why_not_higher": "<为什么不能判更高标签>",
    "why_not_lower": "<为什么不能判更低标签>"
}}"""

        try:
            response = self.llm.chat(
                system_prompt="你是首席安全分析师，只输出JSON。必须保守判断，证据不足时宁可降级。",
                user_prompt=prompt
            )
            
            # 提取JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            result = json.loads(response.strip())
            
            # 校验label范围
            label = int(result.get("label", 2))
            label = max(1, min(5, label))
            result["label"] = label
            
            return result
            
        except Exception as e:
            # LLM失败时降级到概率最大假设
            top_h = max(hypotheses.values(), key=lambda x: x.posterior_probability)
            label_map = {
                "false_positive": 1, "suspected_attack": 2,
                "attack_blocked": 3, "attack_succeeded_not_compromised": 4,
                "compromised": 5
            }
            label = label_map.get(top_h.category, 2)
            return {
                "label": label,
                "label_name": top_h.category,
                "confidence": top_h.posterior_probability,
                "reasoning": f"LLM裁决失败({e})，降级到贝叶斯最优假设: {top_h.name}",
                "attack_chain": [],
                "key_evidence": [],
                "uncertainties": ["LLM裁决异常，结果可能不准确"],
                "why_not_higher": "LLM异常",
                "why_not_lower": "LLM异常"
            }

# ==================== 5. 使用示例 ====================

if __name__ == "__main__":
    import json
    
    # 1. 加载数据
    with open("example.json", encoding='utf-8') as f:
        ndr_data = json.load(f)
    
    # 2. 初始化LLM
    adapter = ChatOpenAIAdapter(llm)
    
    # 3. 意图理解 —— LLM分析NDR图
    engine = IntentUnderstandingEngine(llm_client=adapter)
    structured = engine.parse(ndr_data)  # dict输入，自动走NDR解析
    
    # 4. 假设管理 —— LLM评估所有证据
    hm = HypothesisManager(llm_client=adapter)
    hm.initialize_from_alert(structured)
    
    # 5. 查看LLM分析结果
    print("=== LLM分析的原子事实（前10条）===")
    for fact in structured.atomic_facts[:10]:
        print(f"  {fact}")
    
    # 6. LLM最终裁决
    if hm.judge_engine:
        judgment = hm.judge_engine.adjudicate(
            hm.hypotheses, structured, hm.investigation_history
        )
        print(f"\n=== LLM最终裁决 ===")
        print(f"标签: {judgment['label']} ({judgment['label_name']})")
        print(f"置信度: {judgment['confidence']:.2%}")
        print(f"推理: {judgment['reasoning'][:200]}...")
        print(f"攻击链: {len(judgment['attack_chain'])} 步")
        print(f"不确定性: {judgment['uncertainties']}")


        
            
