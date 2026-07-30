"""
意图理解模块 (Intent Understanding Module)
职责：将原始安全告警解析为结构化的"可推理对象"，作为假设管理引擎的输入。
"""
from enum import Enum
from pydantic import BaseModel, Field
import json
import re,os
from typing import Any, List, Optional, Literal, Union
from datetime import datetime
from dotenv import load_dotenv

# LLM SDK 为可选依赖，纯 JSON 规范化与离线测试无需安装它。
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None


load_dotenv()


def _create_default_llm():
    """仅在 SDK 可用时创建默认客户端，避免导入模块即依赖外部环境。"""
    if ChatOpenAI is None:
        return None
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        request_timeout=60,
        max_retries=3,
    )


llm = _create_default_llm()


class ChatOpenAIAdapter:
    """将 LangChain ChatOpenAI 适配为具有 chat 方法的接口。"""
    def __init__(self, llm: Any):
        if SystemMessage is None or HumanMessage is None:
            raise RuntimeError("使用 ChatOpenAIAdapter 需要安装 langchain-openai 和 langchain-core。")
        self.llm = llm

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        try:
            response = self.llm.invoke(messages)
            # 调试：检查响应结构
            if not response or not response.content:
                print(f"[DEBUG] LLM返回空响应: {response}")
            return response.content or ""
        except Exception as e:
            print(f"[DEBUG] LLM调用异常: {type(e).__name__}: {e}")
            raise


class EntityType(str, Enum):
    IP = "ip"
    DOMAIN = "domain"
    FILE = "file"
    PROCESS = "process"
    USER = "user"
    HASH = "hash"           # MD5/SHA256等
    PORT = "port"
    URL = "url"
    HOSTNAME = "hostname"

class AlertEntity(BaseModel):
    """从告警中提取实体,并进行类型约束"""
    value: str = Field(..., description="实体值，如 192.168.1.1")
    type: EntityType  = Field(..., description="实体类型")
    role: Literal["attacker","victim","intermediate","unknown"] = Field(
        default="unknown", description="实体在该告警中扮演的角色"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="抽取置信度"
    )
    context: Optional[str] = Field(
        default=None, description="该实体在原始日志中的上下文片段"
    )

class AlertSemantics(BaseModel):
    """告警语义分类"""
    category: Literal[
        "malware", "intrusion", "lateral_movement", "exfiltration",
        "reconnaissance", "privilege_escalation", "persistence",
        "defense_evasion", "credential_access", "unknown"
    ] =  Field(default="unknown", description="MITRE ATT&CK 大类映射")
    #tactic：战术
    tactic: Optional[str] = Field(
        default=None, description="具体战术，如 'Brute Force'"
    )
    severity: Literal["critical", "high", "medium", "low", "info"] = Field(
        default="medium", description="严重级别"
    )
    intent_tags: List[str] = Field(
        default_factory=list,
        description="意图标签，如 ['brute_force', 'external_access', 'suspicious_login']"
    )

class StructuredAlert(BaseModel):
    """意图理解模块的最终输出 —— 结构化的告警对象"""
    alert_id: str = Field(..., description="告警唯一ID")
    raw_alert: str = Field(..., description="原始告警文本/日志")
    timestamp: Optional[datetime] = Field(default=None)
    source_system: Optional[str] = Field(
        default=None, description="来源系统，如 WAF/IDS/SIEM/EDR"
    )

    #核心输出
    entities: List[AlertEntity] = Field(
        default_factory=list, description="提取所有实体"
    )
    semantics: AlertSemantics = Field(
        default_factory=AlertSemantics, description="语义分类结果"
    )
    # 用于假设引擎的“原子命题”列表
    atomic_facts: List[str] = Field(
        default_factory=list,
        description="可供假设引擎直接使用的原子事实，如 '源IP为192.168.1.1'"
    )
    # 信息缺口（Information Gap）—— 告诉假设引擎"我还缺什么"
    information_gaps: List[str] = Field(
        default_factory=list,
        description="当前告警中缺失的关键信息，如 '缺少进程链信息'"
    )
    # 原始文本中无法结构化但可能重要的片段
    unstructured_notes: Optional[str] = Field(default=None)


# ==================== 2. 基于规则的实体预抽取（辅助LLM） ====================
class RuleBasedEntityExtractor:
        """
        规则预抽取器：用正则快速提取常见实体，降低LLM负担，同时提供置信度基准。
        注意：这不是替代LLM，而是给LLM提供"候选实体"，让LLM做验证和补充。
        """
        PATTERNS = {
            EntityType.IP: re.compile(
            r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
            ),
            EntityType.HASH: re.compile(
                r'\b(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b'
            ),
            EntityType.DOMAIN: re.compile(
                r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
            ),
            EntityType.URL: re.compile(
                r'https?://[^\s]+'
            ),
            EntityType.PORT: re.compile(
                r'port[:\s]+(\d{1,5})\b|\b:(\d{1,5})\b'
            ),
        }

        def extract(self, text: str) -> List[AlertEntity]:
            entities = []
            seen = set()
            for entity_type, pattern in self.PATTERNS.items():
                for match in pattern.finditer(text):
                    value = match.group(0)
                    # 简单去重
                    if value.lower() in seen:
                        continue
                    seen.add(value.lower())

                    # 简单角色推断
                    role = self._infer_role(text, value, entity_type)
                    entities.append(AlertEntity(
                        value=value,
                        type=entity_type,
                        role=role,
                        confidence=0.7, # 规则抽取的默认置信度
                        context=text[max(0, match.start()-20):match.end()+20]
                    ))
            return entities 

        def _infer_role(self, text: str, value: str, entity_type: EntityType) -> str:
            """基于关键词的简单角色推断"""   
            text_lower = text.lower()
            value_lower = value.lower()

            #源/目的关键词推断
            if entity_type == EntityType.IP:
                if any(k in text_lower[:text_lower.find(value_lower)] for k in ['src', 'source', 'from', '源']) :
                    return "attacker"
                if any(k in text_lower[:text_lower.find(value_lower)] for k in ['dst', 'dest', 'target', 'to', '目的']):
                    return "victim"
            return "unknown"

# ==================== 3. LLM驱动的意图理解核心 ====================


class IntentUnderstandingEngine:

    """
    意图理解引擎
    职责：调用LLM将原始告警解析为StructuredAlert
    """
    
    SYSTEM_PROMPT = """你是一名安全告警分析专家。你的任务是将原始安全告警解析为结构化的分析对象。

## 你的工作流程：
1. 从告警文本中提取所有安全相关实体（IP、域名、文件路径、进程名、用户名、哈希值等）
2. 判断每个实体在该告警中扮演的角色（攻击者/受害者/中间节点/未知）
3. 对告警进行语义分类（映射到MITRE ATT&CK战术类别）
4. 识别信息缺口（当前告警中缺少什么关键信息会导致无法研判？）
5. 生成原子事实列表（供后续假设引擎使用的结构化命题）

## 输出格式要求：
你必须严格按照以下JSON Schema输出，不要添加任何解释：

{
    "entities": [
        {
            "value": "实体值",
            "type": "ip|domain|file|process|user|hash|port|url|hostname",
            "role": "attacker|victim|intermediate|unknown",
            "confidence": 0.95,
            "context": "该实体在原文中的上下文片段"
        }
    ],
    "semantics": {
        "category": "malware|intrusion|lateral_movement|exfiltration|reconnaissance|privilege_escalation|persistence|defense_evasion|credential_access|unknown",
        "tactic": "具体的MITRE战术名称或描述",
        "severity": "critical|high|medium|low|info",
        "intent_tags": ["标签1", "标签2"]
    },
    "atomic_facts": [
        "事实1：源IP为xxx",
        "事实2：触发了xxx规则"
    ],
    "information_gaps": [
        "缺少进程链信息",
        "缺少用户上下文"
    ],
    "unstructured_notes": "任何无法结构化但重要的观察"
}

## 重要原则：
- 只提取告警文本中**明确出现**或**可以合理推断**的信息，不要编造
- 对不确定的信息，降低confidence值
- severity判断要保守：如果没有明确的恶意指标，宁可判为medium也不要判为critical
- information_gaps要具体，指出"缺什么"而不是"需要更多调查"
"""
    def __init__(self, llm_client=None):
        """
        llm_client: 你的LLM客户端，需要实现 `chat(system_prompt, user_prompt) -> str` 接口
        如果为None，则使用模拟模式（用于测试）
        """
        
        self.llm = llm_client
        self.rule_extractor = RuleBasedEntityExtractor()

    def parse(self, raw_alert: Union[str, dict, object], alert_id: str = None,
              source_system: str = None, timestamp: datetime = None) -> StructuredAlert:
        """主入口：安全区分文本、NDR JSON 和未知 JSON。"""
        alert_id = alert_id or f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        observed_at = timestamp or datetime.now()

        if isinstance(raw_alert, dict):
            ndr_keys = {"vertices", "main_edges", "evidences"}
            if ndr_keys.intersection(raw_alert):
                # 仅具备 NDR 特征时才进入专用解析器，避免误解析未知 JSON。
                from GraphParser import NDRGraphParser
                return NDRGraphParser(raw_alert).to_structured_alert()
            return self._unknown_json_alert(raw_alert, alert_id, observed_at)

        if not isinstance(raw_alert, str):
            return self._unknown_json_alert(
                raw_alert,
                alert_id,
                observed_at,
                "输入不是文本或 JSON 对象，未执行攻击研判。",
            )
        if not raw_alert.strip():
            return self._unknown_json_alert(
                raw_alert,
                alert_id,
                observed_at,
                "输入文本为空，缺少可供研判的证据。",
            )

        # Step 1: 规则预抽取（给LLM提供候选）
        rule_entities = self.rule_extractor.extract(raw_alert)
        user_prompt = self._build_prompt(raw_alert, rule_entities)

        if self.llm:
            llm_response = self.llm.chat(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            parsed = self._safe_parse_json(llm_response)
        else:
            # 模拟模式：基于规则生成最小结构化输出
            parsed = self._mock_parse(raw_alert, rule_entities)

        final_entities = self._merge_entities(rule_entities, parsed.get("entities", []))
        return StructuredAlert(
            alert_id=alert_id,
            raw_alert=raw_alert,
            timestamp=observed_at,
            source_system=source_system,
            entities=final_entities,
            semantics=AlertSemantics(**parsed.get("semantics", {})),
            atomic_facts=parsed.get("atomic_facts", []),
            information_gaps=parsed.get("information_gaps", []),
            unstructured_notes=parsed.get("unstructured_notes"),
        )

    @staticmethod
    def _unknown_json_alert(
        raw_alert: object,
        alert_id: str,
        timestamp: datetime,
        diagnostic: str = "未知 JSON 结构，未套用 NDR 规则或攻击结论。",
    ) -> StructuredAlert:
        """为未知或无效输入生成保守诊断，避免入口异常或虚构结论。"""
        if isinstance(raw_alert, dict):
            keys = ", ".join(sorted(map(str, raw_alert.keys()))[:10]) or "无顶层字段"
            diagnostic = f"{diagnostic} 顶层字段: {keys}。"
            raw_text = json.dumps(raw_alert, ensure_ascii=False, default=str)[:2000]
        else:
            raw_text = str(raw_alert)[:2000]
        return StructuredAlert(
            alert_id=alert_id,
            raw_alert=raw_text,
            timestamp=timestamp,
            source_system="UNKNOWN_JSON",
            semantics=AlertSemantics(category="unknown", severity="info"),
            information_gaps=["需要确认输入 JSON 的数据源和字段语义"],
            unstructured_notes=diagnostic,
        )
    
    def _build_prompt(self, raw_alert: str, rule_entities: List[AlertEntity]) -> str:
        """构建给LLM的提示词"""
        entity_hints = ""
        if rule_entities:
            entity_hints = "\n## 规则引擎预提取的候选实体（供参考，请验证）：\n"
            for e in rule_entities:
                entity_hints += f"- [{e.type.value}] {e.value} (role={e.role}, conf={e.confidence})\n"
        
        return f"""请分析以下安全告警，并按照系统指令输出JSON。

=== 原始告警 ===
{raw_alert}
{entity_hints}
"""
    def _safe_parse_json(self, text: str) -> dict:
        """安全解析 LLM 返回的对象 JSON，非法根类型统一降级。"""
        if not isinstance(text, str):
            text = ""
        # 尝试提取 JSON 代码块。
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                return parsed
            diagnostic = "LLM输出根类型不是对象，需要人工介入"
        except (json.JSONDecodeError, TypeError):
            diagnostic = "LLM输出解析失败，需要人工介入"

        return {
            "entities": [],
            "semantics": {},
            "atomic_facts": [],
            "information_gaps": [diagnostic],
            "unstructured_notes": f"原始LLM输出：{text[:500]}",
        }
    
    def _mock_parse(self, raw_alert: str, rule_entities: List[AlertEntity]) -> dict:
        """模拟解析（无LLM时的降级方案）"""
        # 基于关键词的简单语义推断
        text_lower = raw_alert.lower()

        category = "unknown"
        if any(k in text_lower for k in ["brute force", "暴力破解", "login fail", "密码错误"]):
            category = "credential_access"
        elif any(k in text_lower for k in ["malware", "病毒", "木马", "恶意软件"]):
            category = "malware"
        elif any(k in text_lower for k in ["lateral", "横向", "内网传播"]):
            category = "lateral_movement"
        elif any(k in text_lower for k in ["scan", "扫描", "probe"]):
            category = "reconnaissance"
        
        severity = "medium"
        if any(k in text_lower for k in ["critical", "严重", "紧急"]):
            severity = "critical"
        elif any(k in text_lower for k in ["high", "高危"]):
            severity = "high"
        
        atomic_facts = [f"告警内容包含：{raw_alert[:100]}..."]
        for e in rule_entities:
            atomic_facts.append(f"检测到{e.type.value}实体：{e.value}")
        
        return {
            "entities": [e.model_dump() for e in rule_entities],
            "semantics": {
                "category": category,
                "tactic": "自动推断",
                "severity": severity,
                "intent_tags": [category] if category != "unknown" else []
            },
            "atomic_facts": atomic_facts,
            "information_gaps": ["缺少关联上下文", "缺少历史行为基线"],
            "unstructured_notes": "此为规则模式降级输出，建议接入LLM以获得更准确结果"
        }
    
    def _merge_entities(self, rule_entities: List[AlertEntity], llm_entities: List[dict]) -> List[AlertEntity]:
        """合并规则抽取和LLM抽取的实体（去重，优先LLM结果）"""
        seen = {}
        # 先放入LLM抽取的（置信度通常更高）
        for e_data in llm_entities:
            try:
                entity = AlertEntity(**e_data)
                key = f"{entity.type.value}:{entity.value.lower()}"
                seen[key] = entity
            except Exception:
                continue
        
        # 再补充规则抽取的(如果LLM没抽到)
        for entity in rule_entities:
            key = f"{entity.type.value}:{entity.value.lower()}"
            if key not in seen:
                seen[key] = entity
        
        return list(seen.values())


if __name__ == "__main__":
    # 示例告警（模拟WAF/IDS告警）
    sample_alert = """
    [CRITICAL] WAF Alert ID: 20240718-001
    Source: 45.33.22.11 (External)
    Target: 192.168.10.50:8080 (Internal Web Server)
    Rule: SQL Injection Attempt Detected
    URL: http://192.168.10.50:8080/api/login?username=admin' OR '1'='1
    Method: POST
    User-Agent: sqlmap/1.7.2
    Timestamp: 2024-07-18T14:32:10Z
    """
    
    # 初始化引擎（无LLM模式，用于测试）
    adapter = ChatOpenAIAdapter(llm)
    engine = IntentUnderstandingEngine(llm_client=adapter)
    
    # 解析告警
    result = engine.parse(
        raw_alert=sample_alert,
        alert_id="ALERT-20240718-001",
        source_system="WAF"
    )
    
    # 打印结果
    print("=" * 60)
    print(f"告警ID: {result.alert_id}")
    print(f"来源系统: {result.source_system}")
    print(f"语义分类: {result.semantics.category} | 严重级别: {result.semantics.severity}")
    print(f"战术: {result.semantics.tactic}")
    print(f"意图标签: {result.semantics.intent_tags}")
    print("-" * 60)
    print("提取实体:")
    for e in result.entities:
        print(f"  [{e.type.value:8}] {e.value:20} role={e.role:12} conf={e.confidence:.2f}")
    print("-" * 60)
    print("原子事实:")
    for fact in result.atomic_facts:
        print(f"  • {fact}")
    print("-" * 60)
    print("信息缺口:")
    for gap in result.information_gaps:
        print(f"  ⚠ {gap}")
    print("=" * 60)
    
    # 输出JSON（供假设引擎消费）
    print("\n=== 输出给假设引擎的JSON ===")
    print(result.model_dump_json(indent=2, ensure_ascii=False))



            