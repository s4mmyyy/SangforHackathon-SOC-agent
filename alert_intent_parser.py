"""
意图理解模块 (Intent Understanding Module)
职责：将原始安全告警解析为结构化的"可推理对象"，作为假设管理引擎的输入。
"""
from enum import Enum
from pydantic import BaseModel, Field
import json
import re
from typing import List, Optional, Literal
from datetime import datetime

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

        def extract(self, text: str) -> self.PATTERNS.items():
            for 