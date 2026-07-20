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

    prior_probability: float = 0.5
    posterior_probability: float = 0.5
    
