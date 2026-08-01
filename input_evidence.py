"""输入泛化与证据保真模块：保留原始 JSON、规范化副本及可追溯证据。"""

from __future__ import annotations

import copy
import hashlib
import html
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import unquote


_PERCENT_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")       # 匹配百分号码
_TIME_SUFFIX = re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]")  # 匹配的时间前缀
_TRUNCATION_MARKERS = ("<TRUNCATED>", "[......]")      # 截断标记，表示值被截断。
_REDACTION_MARKERS = ("<OMITTED>",)                    # 脱敏标记，表示值被省略。
_MAX_PROFILE_DEPTH = 12                                # 递归展开 JSON 的最大深度，防止过度递归。
_MAX_PROFILE_NODES = 2000                              # 最大节点数，防止过大 JSON 耗尽资源。
_MAX_SAMPLE_LENGTH = 160                               # 样本展示的最大长度。


@dataclass
class TransformationRecord:
    """记录单次文本变换，支持审计原文与规范化值的差异。"""

    operation: str   # 操作名称。
    pass_index: int  # 变换轮次（通常为 1，因为每个字符串只做一次）。
    changed: bool    # 是否发生变化。
    input_sha256: str # 变换前字符串的哈希。
    output_sha256: str # 变换后字符串的哈希。


@dataclass
class IntegrityInfo:
    """记录来源文本自身的截断或脱敏标志，不补造缺失内容。"""

    truncated: bool = False
    redacted: bool = False
    markers: List[str] = field(default_factory=list)


@dataclass
class EvidenceRecord:
    """统一证据对象：每项均可回溯至原始 JSONPath。"""

    evidence_id: str
    source_system: str  # 来源系统标识，如 "NDR" 或 "UNKNOWN_JSON"。
    kind: str           # 证据种类（如 json_scalar, ndr_alert_aggregation 等）。
    source_path: str
    raw_value: Any
    normalized_value: Any   # 规范化后的值。
    transformations: List[TransformationRecord] = field(default_factory=list)   # 变换记录列表。
    integrity: IntegrityInfo = field(default_factory=IntegrityInfo) # 完整性信息。
    related_evidence_ids: List[str] = field(default_factory=list)   # 关联的其他证据 ID。
    attributes: Dict[str, Any] = field(default_factory=dict)        # 额外属性，如时间戳、索引等。


@dataclass
class JsonProfileEntry:
    """JSON 结构剖面：只记录形状、样本与候选时间，不产生安全结论。"""

    path: str
    value_type: str
    sample: Optional[str] = None
    length: Optional[int] = None
    array_summary: Optional[Dict[str, Any]] = None  # 数组统计（采样项数、类型计数）。
    time_candidate: Optional[str] = None


@dataclass
class DetectionResult:
    """数据源特征识别结果，仅用于选择适配器。"""

    adapter: str        # 识别的适配器名称（如 "NDR" 或 "UNKNOWN_JSON"）。
    confidence: str     # 置信度（"none", "low", "medium", "high"）。
    matched_features: List[str] = field(default_factory=list)   # 匹配到的特征列表


@dataclass
class InputEvidenceBundle:
    """完整输入包：原始值与规范化值并存，供后续 Agent 与适配器使用。"""

    raw_payload: Any
    normalized_payload: Any
    profile: List[JsonProfileEntry]
    evidence_records: List[EvidenceRecord]
    diagnostics: List[str]
    detection: DetectionResult

# ========== 将证据记录和剖面转换为字典列表，便于序列化（例如给 Pydantic 输出） ==========
    def evidence_as_dicts(self) -> List[Dict[str, Any]]:
        """为现有 Pydantic 输出提供可序列化的证据记录。"""
        return [asdict(record) for record in self.evidence_records]

    def profile_as_dicts(self) -> List[Dict[str, Any]]:
        """为现有 Pydantic 输出提供可序列化的 profile 记录。"""
        return [asdict(entry) for entry in self.profile]


def _sha256(value: Any) -> str: # 对任意值生成稳定 SHA-256 摘要。非字符串先转为 repr()，然后 UTF-8 编码，错误替换
    """对任意值生成稳定摘要，用于记录文本变换前后版本。"""
    if not isinstance(value, str):
        value = repr(value)
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _json_type(value: Any) -> str:
    """将 Python 值映射为 JSON 类型名称。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _sample(value: Any) -> str:
    """生成受限展示样本，完整内容始终保存在 raw_payload/evidence 中。"""
    if isinstance(value, str):
        return value[:_MAX_SAMPLE_LENGTH] # 截断到 _MAX_SAMPLE_LENGTH
    try:
        return repr(value)[:_MAX_SAMPLE_LENGTH]
    except Exception:
        return "<unrenderable>"


def _time_candidate(value: Any) -> Optional[str]:
    """检查字符串是否可能是 ISO 8601 时间（以日期开头），且能解析，则返回原值，不改写来源时间字段。"""
    if not isinstance(value, str) or not _TIME_SUFFIX.match(value):
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
    except ValueError:
        return None


def _integrity(value: Any) -> IntegrityInfo:
    """检测上游声明的截断/脱敏标识，作为证据完整性提示。"""
    if not isinstance(value, str):
        return IntegrityInfo()  # 检测字符串中是否包含截断或脱敏标记，并返回 IntegrityInfo 对象
    markers = [marker for marker in _TRUNCATION_MARKERS + _REDACTION_MARKERS if marker in value]
    return IntegrityInfo(
        truncated=any(marker in value for marker in _TRUNCATION_MARKERS),
        redacted=any(marker in value for marker in _REDACTION_MARKERS),
        markers=markers,
    )


def _is_url_context(path: str, value: str) -> bool:
    """仅识别 URL 值或 HTTP 请求目标，避免将任意正文做百分号解码。"""
    leaf = path.rsplit(".", 1)[-1].lower()
    if leaf in {"url", "uri", "request_target", "entity_value"}:
        return True
    if value.startswith(("http://", "https://", "/")):
        return True
    return False


def _decode_http_request_target(value: str) -> str:
    """只解码 HTTP 请求行中的 target（如 GET /path%20with%20space HTTP/1.1），避免误处理其余请求头或正文。"""
    match = re.match(r"^([A-Z]+\s+)(\S+)(\s+HTTP/\d(?:\.\d)?)", value)
    if not match or not _PERCENT_ESCAPE.search(match.group(2)):
        return value
    return f"{match.group(1)}{unquote(match.group(2))}{match.group(3)}" + value[match.end():]


def _normalize_string(value: str, path: str) -> tuple[str, List[TransformationRecord]]:
    """执行一次实体解码与一次受控 URL 解码，原始值绝不覆盖。"""
    transformations: List[TransformationRecord] = []
    entity_decoded = html.unescape(value)
    transformations.append(TransformationRecord(
        operation="html_xml_entity_decode",  #  先做 HTML/XML 实体解码（如 &amp; → &），记录变换
        pass_index=1,
        changed=entity_decoded != value,
        input_sha256=_sha256(value),
        output_sha256=_sha256(entity_decoded),
    ))

    normalized = entity_decoded
    leaf = path.rsplit(".", 1)[-1].lower()
    if leaf == "request_headers":  # 然后根据路径判断是否进行 URL 百分号解码。特殊处理 request_headers 字段调用专门函数
        url_decoded = _decode_http_request_target(normalized)
    elif _is_url_context(path, normalized) and _PERCENT_ESCAPE.search(normalized):
        url_decoded = unquote(normalized)
    else:
        url_decoded = normalized
    if url_decoded != normalized:
        transformations.append(TransformationRecord(
            operation="url_percent_decode",
            pass_index=1,
            changed=True,
            input_sha256=_sha256(normalized),
            output_sha256=_sha256(url_decoded),
        ))
        normalized = url_decoded
    return normalized, transformations


def _make_evidence_id(path: str, kind: str) -> str:
    """由路径和种类生成稳定 ID，便于跨来源关联。"""
    digest = hashlib.sha256(f"{kind}:{path}".encode("utf-8")).hexdigest()[:16]
    return f"ev_{digest}"


# ========== 主识别函数 ==========
def detect_ndr_payload(payload: Any) -> DetectionResult:
    """根据结构特征识别 NDR，单个同名字段不足以触发专用适配。"""
    if not isinstance(payload, dict): # 检查payload是否为字典类型，如果不是字典，说明不是合法的 JSON 对象，立即返回
        return DetectionResult(adapter="UNKNOWN_JSON", confidence="none")
    vertices = payload.get("vertices")
    edges = payload.get("main_edges")
    if not isinstance(vertices, list) or not isinstance(edges, list): # 检查 vertices 和 edges 是否都是列表类型。如果不是，则说明数据结构不符合基本预期（NDR 需要顶点和边列表）
        return DetectionResult(adapter="UNKNOWN_JSON", confidence="low")

    features = ["vertices:list", "main_edges:list"]
    score = 2 # 初始化得分 score = 2
    if isinstance(payload.get("evidences"), list):
        score += 1      # 检查 payload 中是否存在 "evidences" 键，且其值为列表类型。如果满足，则得分加 1
        features.append("evidences:list")
    if any(isinstance(item, dict) and {"id", "type"}.issubset(item) for item in vertices):
        score += 1      # 如果说明顶点包含必要的标识和类型信息，得分加 1
        features.append("vertex:id,type")
    if any(isinstance(item, dict) and {"src", "dst", "alert_edges"}.issubset(item) for item in edges): # 检查是否存在至少一个元素是字典
        score += 2      # 该字典的键集合包含 {"src", "dst", "alert_edges"} 三个字段。如果存在这样的边，说明边包含源、目的和告警边信息，得分加 2（权重更高）
        features.append("edge:src,dst,alert_edges")
    if score >= 4:
        confidence = "high" if score >= 5 else "medium"   # 判断总得分是否达到 4 分或以上。如果是，则认为是 NDR 格式。进一步根据得分细分置信度：若得分 >= 5，置信度为 "high"；否则（即得分为 4）为 "medium"
        return DetectionResult(adapter="NDR", confidence=confidence, matched_features=features)
    return DetectionResult(adapter="UNKNOWN_JSON", confidence="low", matched_features=features)


def build_input_evidence(payload: Any, source_system: str = "UNKNOWN_JSON") -> InputEvidenceBundle:
    """递归构建输入 profile 与标量证据；原始 payload 仅深拷贝，不做截断。"""
    raw_payload = copy.deepcopy(payload)  # 深拷贝，保留原始数据副本，避免后续处理（如规范化）修改原始输入
    diagnostics: List[str] = []           # 用于记录处理过程中的诊断信息（如超限警告）
    profile: List[JsonProfileEntry] = []  # 用于存储每个路径的结构概要
    evidence_records: List[EvidenceRecord] = [] # 存储每个叶子节点（标量值）的证据记录
    node_count = 0

    def walk(value: Any, path: str, depth: int) -> Any:  # 内部递归函数，value：当前要处理的子数据；path：当前节点在 JSON 中的路径；depth：当前递归深度
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_PROFILE_NODES:
            diagnostics.append(f"JSON profile 超过最大节点数 {_MAX_PROFILE_NODES}，后续路径未展开。")
            return copy.deepcopy(value)
        if depth > _MAX_PROFILE_DEPTH:
            diagnostics.append(f"路径 {path} 超过最大深度 {_MAX_PROFILE_DEPTH}，后续路径未展开。")
            return copy.deepcopy(value)

        value_type = _json_type(value)  # 调用外部函数 _json_type 获取当前值的类型字符串
        if isinstance(value, dict):  # 处理字典类型
            profile.append(JsonProfileEntry(path=path, value_type=value_type, length=len(value)))
            normalized: Dict[str, Any] = {}
            for key, child in value.items():   # 遍历原字典的每个键值对，如果键不是字符串，则记录一条诊断，并将键转换为字符串，这里是为了python做兼容处理
                if not isinstance(key, str):
                    diagnostics.append(f"路径 {path} 包含非字符串键，已转换为展示键。")
                    key = str(key)
                normalized[key] = walk(child, f"{path}.{key}", depth + 1)
            return normalized

        if isinstance(value, list):  
            type_counts: Dict[str, int] = {}
            for child in value[:20]:
                child_type = _json_type(child)
                type_counts[child_type] = type_counts.get(child_type, 0) + 1
            profile.append(JsonProfileEntry(
                path=path,
                value_type=value_type,
                length=len(value),
                array_summary={"sampled_items": min(len(value), 20), "type_counts": type_counts},
            ))
            if not value:
                diagnostics.append(f"路径 {path} 是空数组。")
            return [walk(child, f"{path}[{index}]", depth + 1) for index, child in enumerate(value)]

# ========== 处理标量值（非 dict 或 list，如字符串、数字、布尔值、None） ==========
        normalized_value = value   
        transformations: List[TransformationRecord] = []
        if isinstance(value, str):
            normalized_value, transformations = _normalize_string(value, path)  # 如果值是字符串，调用外部函数 _normalize_string 进行规范化（例如去除空白、格式化日期等），返回规范化后的字符串和转换记录列表。
        profile.append(JsonProfileEntry(
            path=path,
            value_type=value_type,
            sample=_sample(value),
            length=len(value) if isinstance(value, (str, list, dict)) else None,
            time_candidate=_time_candidate(value),
        ))
        evidence_records.append(EvidenceRecord(  # 向 evidence_records 添加一条证据记录
            evidence_id=_make_evidence_id(path, "json_scalar"),
            source_system=source_system,
            kind="json_scalar",
            source_path=path,
            raw_value=value,
            normalized_value=normalized_value,
            transformations=transformations,
            integrity=_integrity(value),
        ))
        return normalized_value  # 返回规范化后的标量值

    normalized_payload = walk(raw_payload, "$", 0)
    detection = detect_ndr_payload(raw_payload)

# 如果识别为 NDR，调用 enrich_ndr_evidence 补充 NDR 特有的证据
    if detection.adapter == "NDR":
        enrich_ndr_evidence(raw_payload, normalized_payload, evidence_records, diagnostics)
    return InputEvidenceBundle(
        raw_payload=raw_payload,
        normalized_payload=normalized_payload,
        profile=profile,
        evidence_records=evidence_records,
        diagnostics=diagnostics,
        detection=detection,
    )
# ========== 增强函数 ==========
def enrich_ndr_evidence(
    raw_payload: Any,
    normalized_payload: Any,                # 经过 build_input_evidence 规范化后的数据
    evidence_records: List[EvidenceRecord], # 证据记录列表（可变，函数会向其追加新记录）
    diagnostics: List[str],                 # 诊断信息列表（可变，可添加警告或错误信息）
) -> None:
    """补充 NDR 专用证据，保留 HTTP、聚合语义和 evidence_ids 引用关系。"""
    if not isinstance(raw_payload, dict) or not isinstance(normalized_payload, dict): # 检查数据是否为字典类型
        return

    registry: Dict[str, str] = {}
    raw_registry = raw_payload.get("evidences", [])
    normalized_registry = normalized_payload.get("evidences", [])
    if isinstance(raw_registry, list) and isinstance(normalized_registry, list):
        for index, raw_item in enumerate(raw_registry):  # 遍历 raw_registry 列表，使用 enumerate 同时获取索引 index 和元素 raw_item
            if not isinstance(raw_item, dict):
                continue
            normalized_item = normalized_registry[index] if index < len(normalized_registry) else raw_item # 如果索引 index 在 normalized_registry 范围内，则取 normalized_registry[index]，否则回退为 raw_item
            source_id = raw_item.get("evidence_id") or raw_item.get("id") or raw_item.get("entity_value") # 获取标识符，"evidence_id"或"id"或"entity_value"
            path = f"$.evidences[{index}]" # 构造当前证据条目在 JSON 中的路径字符串
            record = EvidenceRecord(  # 创建一个新的 EvidenceRecord
                evidence_id=_make_evidence_id(path, "ndr_registry"),
                source_system="NDR",
                kind="ndr_evidence_registry",
                source_path=path,
                raw_value=raw_item,
                normalized_value=normalized_item,
                attributes={"source_evidence_id": source_id, "entity_type": raw_item.get("type")},
            )
            evidence_records.append(record)
            if isinstance(source_id, str) and source_id:   # 如果 source_id 是非空字符串，则在 registry 字典中建立映射：键为 source_id，值为新记录的证据 ID
                registry[source_id] = record.evidence_id

    raw_edges = raw_payload.get("main_edges", [])
    normalized_edges = normalized_payload.get("main_edges", [])
    if not isinstance(raw_edges, list) or not isinstance(normalized_edges, list):
        return
    for edge_index, raw_edge in enumerate(raw_edges):  # 遍历原始边列表，使用 enumerate 获取索引 edge_index 和边对象 raw_edge
        if not isinstance(raw_edge, dict):
            continue
        normalized_edge = normalized_edges[edge_index] if edge_index < len(normalized_edges) else raw_edge
        raw_alert_edges = raw_edge.get("alert_edges", [])
        normalized_alert_edges = normalized_edge.get("alert_edges", []) if isinstance(normalized_edge, dict) else []
        if not isinstance(raw_alert_edges, list):
            continue
        for alert_index, raw_alert_edge in enumerate(raw_alert_edges):
            if not isinstance(raw_alert_edge, dict):
                continue
            normalized_alert_edge = (
                normalized_alert_edges[alert_index]
                if isinstance(normalized_alert_edges, list) and alert_index < len(normalized_alert_edges)
                else raw_alert_edge
            )
            base_path = f"$.main_edges[{edge_index}].alert_edges[{alert_index}]"  # 构造当前告警边的基础路径，例如 "$.main_edges[0].alert_edges[2]"
            related_ids: List[str] = []
            source_ids = raw_alert_edge.get("evidence_ids", [])
            if isinstance(source_ids, list):
                for source_id in source_ids:
                    if isinstance(source_id, str) and source_id in registry:
                        related_ids.append(registry[source_id])
                    elif isinstance(source_id, str):
                        diagnostics.append(f"路径 {base_path}.evidence_ids 引用了未登记证据 {source_id}。")

            aggregation_fields = { # 构造一个字典 aggregation_fields，从原始告警边中提取指定字段
                key: raw_alert_edge.get(key)
                for key in ("alert_vid", "ts", "count", "foldedStatement", "folded_alert_vids", "evidence_ids")
                if key in raw_alert_edge
            }
            evidence_records.append(EvidenceRecord( 
                evidence_id=_make_evidence_id(base_path, "ndr_alert_aggregation"),
                source_system="NDR",
                kind="ndr_alert_aggregation",
                source_path=base_path,
                raw_value=aggregation_fields,
                normalized_value={
                    key: normalized_alert_edge.get(key)
                    for key in aggregation_fields
                    if isinstance(normalized_alert_edge, dict)
                },
                related_evidence_ids=related_ids,
                attributes={"edge_index": edge_index, "alert_index": alert_index},
            ))

            raw_alert = raw_alert_edge.get("alert", {})
            normalized_alert = normalized_alert_edge.get("alert", {}) if isinstance(normalized_alert_edge, dict) else {}
            if not isinstance(raw_alert, dict):
                continue
            for field_name, kind in (  # 定义一个元组列表，遍历四种 HTTP 相关的字段及其对应的证据种类名称
                ("request_headers", "ndr_http_request_headers"),
                ("request_body", "ndr_http_request_body"),
                ("response_headers", "ndr_http_response_headers"),
                ("response_body", "ndr_http_response_body"),
            ):
                if field_name not in raw_alert:
                    continue
                path = f"{base_path}.alert.{field_name}"
                raw_value = raw_alert.get(field_name)
                normalized_value = normalized_alert.get(field_name) if isinstance(normalized_alert, dict) else raw_value
                # 通用标量记录已存在；这里用专用 kind 增加 HTTP 语义和聚合关联。
                evidence_records.append(EvidenceRecord(
                    evidence_id=_make_evidence_id(path, kind),
                    source_system="NDR",
                    kind=kind,
                    source_path=path,
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                    integrity=_integrity(raw_value),
                    related_evidence_ids=related_ids,
                    attributes={"alert_vid": raw_alert_edge.get("alert_vid"), "timestamp": raw_alert_edge.get("ts")},
                ))


# 阶段 2 仅允许模型通过以下只读工具查看当前输入包，禁止任意路径执行。
_SAFE_PATH = re.compile(r"^\$(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*$")


def _find_profile(bundle: InputEvidenceBundle, path: str) -> Optional[JsonProfileEntry]:
    """只允许读取 profile 中已存在的安全 JSONPath。"""
    if not isinstance(path, str) or not _SAFE_PATH.fullmatch(path):
        return None
    return next((entry for entry in bundle.profile if entry.path == path), None)


def _value_at_path(value: Any, path: str) -> Any:  # 根据安全路径从规范化载荷中提取实际值
    """按已校验的简化 JSONPath 读取规范化副本，绝不执行表达式。"""
    if path == "$":
        return value
    cursor = value
    for key, index in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", path):
        try:
            cursor = cursor[key] if key else cursor[int(index)]
        except (KeyError, IndexError, TypeError):
            return None
    return cursor


def inspect_json_structure(
    bundle: InputEvidenceBundle,
    path: str = "$",
    max_children: int = 20,
    max_array_items: int = 10,
) -> Dict[str, Any]:
    """返回受限 JSON 结构视图，不泄露任意大字段或执行路径表达式。"""
    profile_entry = _find_profile(bundle, path)
    if profile_entry is None:
        return {
            "tool": "inspect_json_structure",
            "status": "rejected",
            "reason_code": "PATH_NOT_PROFILED",
            "path": path,
            "available_next_actions": ["inspect_json_structure", "finish"],
        }
    max_children = max(1, min(int(max_children), 30))  # 限制参数范围，防止模型请求过大数据导致输出过大
    max_array_items = max(1, min(int(max_array_items), 10))
    value = _value_at_path(bundle.normalized_payload, path)  # 获取值并构建响应
    response: Dict[str, Any] = {
        "tool": "inspect_json_structure",
        "status": "ok",
        "path": path,
        "value_type": profile_entry.value_type,
        "length": profile_entry.length,
        "matching_profile_paths": [
            entry.path for entry in bundle.profile if entry.path == path or entry.path.startswith(path + ".") or entry.path.startswith(path + "[")
        ][:40],
        "diagnostics": bundle.diagnostics[:20],
    }
    if isinstance(value, dict):  # 字典：显示前 max_children 个子节点，每个包含子路径和类型
        response["children"] = [
            {"path": f"{path}.{key}", "value_type": _json_type(child)}
            for key, child in list(value.items())[:max_children]
            if isinstance(key, str)
        ]
    elif isinstance(value, list):  # 列表：显示前 max_array_items 个元素，每个包含索引路径和类型
        response["children"] = [
            {"path": f"{path}[{index}]", "value_type": _json_type(child)}
            for index, child in enumerate(value[:max_array_items])
        ]
    else:
        response["sample"] = _sample(value)  # 标量值：显示受限样本（截断到 _MAX_SAMPLE_LENGTH）
    matching_records = [
        record
        for record in bundle.evidence_records
        if record.source_path == path or record.source_path.startswith(path + ".") or record.source_path.startswith(path + "[")
    ][:30]
    response["evidence_ids"] = [record.evidence_id for record in matching_records]
    response["evidence_references"] = [
        {
            "evidence_id": record.evidence_id,
            "source_path": record.source_path,
            "kind": record.kind,
        }
        for record in matching_records
    ]
    return response


def inspect_evidence(  # 按已分配的证据ID返回具体证据的详细内容
    bundle: InputEvidenceBundle,
    evidence_ids: List[str],
    max_value_chars: int = 1000,
) -> Dict[str, Any]:
    """按已分配 ID 返回有限证据展示；内容始终标注为不可信观察。"""
    if not isinstance(evidence_ids, list) or not evidence_ids or len(evidence_ids) > 20:
        return {
            "tool": "inspect_evidence",
            "status": "rejected",
            "reason_code": "EVIDENCE_ID_LIMIT_INVALID",
            "evidence_ids": [],
        }
    max_value_chars = max(100, min(int(max_value_chars), 2000))
    registry = {record.evidence_id: record for record in bundle.evidence_records}  # 构建证据注册表  将 evidence_records 转换为字典，键为 evidence_id，方便快速查找
    records = []
    missing = []
    for evidence_id in evidence_ids:  # 遍历证据ID
        record = registry.get(evidence_id)
        if record is None:
            missing.append(evidence_id)
            continue
        records.append({
            "evidence_id": record.evidence_id,
            "source_path": record.source_path,
            "kind": record.kind,
            "untrusted_raw_preview": _sample(record.raw_value)[:max_value_chars],
            "untrusted_normalized_preview": _sample(record.normalized_value)[:max_value_chars],
            "transformations": [asdict(item) for item in record.transformations],
            "integrity": asdict(record.integrity),
            "related_evidence_ids": record.related_evidence_ids,
        })
    status = "ok" if records else "empty"
    return {
        "tool": "inspect_evidence",
        "status": status,
        "evidence_ids": [record["evidence_id"] for record in records],
        "records": records,
        "missing_evidence_ids": missing,
        "reason_code": "EVIDENCE_NOT_FOUND" if missing and not records else None,
    }
