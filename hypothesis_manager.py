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