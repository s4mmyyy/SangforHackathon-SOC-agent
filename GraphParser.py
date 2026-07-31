"""NDR 图输入适配器：保守抽取可观察证据，并记录结构诊断。"""

from __future__ import annotations

import collections
"""NDR 图输入适配器：保守抽取可观察证据，并记录结构诊断。"""

from __future__ import annotations

import collections
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from alert_intent_parser import AlertEntity, AlertSemantics, EntityType, StructuredAlert
from input_evidence import InputEvidenceBundle, build_input_evidence


class NDRGraphParser:
    """将 NDR JSON 转换为统一告警，不在此层生成攻击成功结论。"""

    _ROOT_FIELDS = {"tenant", "diffused_at", "vertices", "main_edges", "evidences"}

    def __init__(
        self,
        ndr_json: object,
        llm_client: object = None,
        evidence_bundle: InputEvidenceBundle | None = None,
    ):
        # 图计算只使用规范化副本；完整原始输入由 evidence_bundle 持续保留。
        self.llm = llm_client
        self.bundle = evidence_bundle or build_input_evidence(ndr_json, source_system="NDR")
        # NDR 适配成功后统一标记来源，便于下游按数据源筛选证据。
        for record in self.bundle.evidence_records:
            record.source_system = "NDR"
        self.diagnostics: List[str] = list(self.bundle.diagnostics)
        self.data = self._normalize_payload(self.bundle.normalized_payload)
        self.vertices = self._normalize_vertices(self.data["vertices"])
        self.edges = self._normalize_edges(self.data["main_edges"])
        self.evidences = self._normalize_object_list(self.data["evidences"], "evidences")

    def _normalize_payload(self, payload: object) -> Dict[str, Any]:
        """规范化根对象，任何异常输入都转换为可诊断的空图。"""
        if not isinstance(payload, dict):
            self.diagnostics.append("NDR 输入根节点不是 JSON 对象，已按空图处理。")
            payload = {}
        elif not payload:
            self.diagnostics.append("NDR 输入对象为空，未发现可解析字段。")

        unknown_fields = sorted(set(payload) - self._ROOT_FIELDS)
        if unknown_fields:
            self.diagnostics.append(f"发现未知顶层字段：{', '.join(map(str, unknown_fields))}。")

        normalized: Dict[str, Any] = {
            "tenant": payload.get("tenant", "UNKNOWN"),
            "diffused_at": payload.get("diffused_at", ""),
        }
        for field_name in ("vertices", "main_edges", "evidences"):
            value = payload.get(field_name, [])
            if field_name not in payload:
                self.diagnostics.append(f"缺少顶层字段 {field_name}，已使用空列表。")
            elif not isinstance(value, list):
                self.diagnostics.append(f"顶层字段 {field_name} 不是数组，已使用空列表。")
                value = []
            elif not value:
                self.diagnostics.append(f"顶层字段 {field_name} 为空数组。")
            normalized[field_name] = value
        return normalized

    def _normalize_object_list(self, value: object, field_name: str) -> List[Dict[str, Any]]:
        """过滤非对象元素，避免后续字段访问抛出类型异常。"""
        if not isinstance(value, list):
            self.diagnostics.append(f"字段 {field_name} 不是数组，已忽略。")
            return []
        result: List[Dict[str, Any]] = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                result.append(item)
            else:
                self.diagnostics.append(f"字段 {field_name}[{index}] 不是对象，已忽略。")
        return result

    def _normalize_vertices(self, raw_vertices: object) -> Dict[str, Dict[str, Any]]:
        """校验节点关键字段，缺失节点不参与实体与关系解析。"""
        vertices: Dict[str, Dict[str, Any]] = {}
        for index, vertex in enumerate(self._normalize_object_list(raw_vertices, "vertices")):
            vertex_id = vertex.get("id")
            vertex_type = vertex.get("type")
            if not isinstance(vertex_id, str) or not vertex_id:
                self.diagnostics.append(f"vertices[{index}] 缺少有效 id，已忽略。")
                continue
            if not isinstance(vertex_type, str) or not vertex_type:
                self.diagnostics.append(f"vertices[{index}] 缺少有效 type，已忽略。")
                continue
            properties = vertex.get("properties", {})
            if not isinstance(properties, dict):
                self.diagnostics.append(f"vertices[{index}].properties 不是对象，已使用空对象。")
                properties = {}
            vertices[vertex_id] = {**vertex, "properties": properties}
        if not vertices:
            self.diagnostics.append("没有可用顶点，无法建立实体映射。")
        return vertices

    def _normalize_edges(self, raw_edges: object) -> List[Dict[str, Any]]:
        """校验边与告警数组，保留可观察字段并记录无效引用。"""
        edges: List[Dict[str, Any]] = []
        for index, edge in enumerate(self._normalize_object_list(raw_edges, "main_edges")):
            src = edge.get("src")
            dst = edge.get("dst")
            if not isinstance(src, str) or not src or not isinstance(dst, str) or not dst:
                self.diagnostics.append(f"main_edges[{index}] 缺少有效 src/dst，已忽略。")
                continue
            if src not in self.vertices or dst not in self.vertices:
                self.diagnostics.append(f"main_edges[{index}] 引用了未识别顶点：{src} -> {dst}。")
            alert_edges = edge.get("alert_edges", [])
            if not isinstance(alert_edges, list):
                self.diagnostics.append(f"main_edges[{index}].alert_edges 不是数组，已使用空数组。")
                alert_edges = []
            valid_alert_edges: List[Dict[str, Any]] = []
            for alert_index, alert_edge in enumerate(alert_edges):
                if not isinstance(alert_edge, dict):
                    self.diagnostics.append(
                        f"main_edges[{index}].alert_edges[{alert_index}] 不是对象，已忽略。"
                    )
                    continue
                alert = alert_edge.get("alert", {})
                if not isinstance(alert, dict):
                    self.diagnostics.append(
                        f"main_edges[{index}].alert_edges[{alert_index}].alert 不是对象，已忽略。"
                    )
                    continue
                valid_alert_edges.append({**alert_edge, "alert": alert})
            edges.append({**edge, "alert_edges": valid_alert_edges})
        if not edges:
            self.diagnostics.append("没有可用攻击流，无法形成网络行为时间线。")
        return edges

    @staticmethod
    def _map_vertex_type(value: object) -> EntityType:
        """将来源节点类型映射为统一实体枚举。"""
        mapping = {
            "IP": EntityType.IP,
            "DOMAIN": EntityType.DOMAIN,
            "FILE": EntityType.FILE,
            "PROCESS": EntityType.PROCESS,
            "USER": EntityType.USER,
        }
        return mapping.get(str(value).upper(), EntityType.HASH)

    @staticmethod
    def _vertex_value(vertex_id: str, vertex: Dict[str, Any]) -> str:
        """优先使用节点属性中的 IP，缺失时保留来源节点标识。"""
        properties = vertex.get("properties", {})
        ip_value = properties.get("ip") if isinstance(properties, dict) else None
        return str(ip_value or vertex_id)

    def extract_entities(self) -> List[AlertEntity]:
        """从已验证顶点提取实体，重复值只保留一次。"""
        entities: List[AlertEntity] = []
        seen = set()
        for vertex_id, vertex in self.vertices.items():
            value = self._vertex_value(vertex_id, vertex)
            entity_type = self._map_vertex_type(vertex.get("type"))
            key = (entity_type.value, value.lower())
            if key in seen:
                continue
            seen.add(key)
            role = str(vertex.get("role", "unknown"))
        """从已验证顶点提取实体，重复值只保留一次。"""
        entities: List[AlertEntity] = []
        seen = set()
        for vertex_id, vertex in self.vertices.items():
            value = self._vertex_value(vertex_id, vertex)
            entity_type = self._map_vertex_type(vertex.get("type"))
            key = (entity_type.value, value.lower())
            if key in seen:
                continue
            seen.add(key)
            role = str(vertex.get("role", "unknown"))
            entities.append(AlertEntity(
                value=value,
                type=entity_type,
                role=role if role in {"attacker", "victim", "intermediate"} else "unknown",
                role=role if role in {"attacker", "victim", "intermediate"} else "unknown",
                confidence=1.0,
                context=json.dumps(vertex.get("properties", {}), ensure_ascii=False),
                context=json.dumps(vertex.get("properties", {}), ensure_ascii=False),
            ))
        return entities

    @staticmethod
    def _extract_status_code(headers: object) -> int:
        """安全提取响应状态码；未识别时返回 0。"""
        if not isinstance(headers, str):
            return 0
        match = re.search(r"HTTP/\d(?:\.\d)?\s+(\d+)", headers)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _endpoint_label(endpoint: object) -> str:
        """将边端点转为展示文本，不假设其为 IP。"""
        value = str(endpoint or "unknown")
        return value.split(":", 1)[-1] if ":" in value else value

    def generate_atomic_facts(self, max_facts_per_edge: int = 3) -> List[str]:
        """生成观察事实，不基于关键词或状态码推导攻击成败。"""
        facts: List[str] = []

    @staticmethod
    def _extract_status_code(headers: object) -> int:
        """安全提取响应状态码；未识别时返回 0。"""
        if not isinstance(headers, str):
            return 0
        match = re.search(r"HTTP/\d(?:\.\d)?\s+(\d+)", headers)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _endpoint_label(endpoint: object) -> str:
        """将边端点转为展示文本，不假设其为 IP。"""
        value = str(endpoint or "unknown")
        return value.split(":", 1)[-1] if ":" in value else value

    def generate_atomic_facts(self, max_facts_per_edge: int = 3) -> List[str]:
        """生成观察事实，不基于关键词或状态码推导攻击成败。"""
        facts: List[str] = []
        attackers = [v for v in self.vertices.values() if v.get("role") == "attacker"]
        victims = [v for v in self.vertices.values() if v.get("role") == "victim"]
        facts.append(
            f"全局拓扑观察：攻击者节点={len(attackers)}，受害节点={len(victims)}，可用攻击流={len(self.edges)}。"
        )
        if not self.edges:
            facts.append("未观察到可用攻击流，不能据此判断攻击是否发生或成功。")
            return facts

        facts.append(
            f"全局拓扑观察：攻击者节点={len(attackers)}，受害节点={len(victims)}，可用攻击流={len(self.edges)}。"
        )
        if not self.edges:
            facts.append("未观察到可用攻击流，不能据此判断攻击是否发生或成功。")
            return facts

        for edge in self.edges:
            threat_types = sorted({
                str(item["alert"].get("threat_type"))
                for item in edge["alert_edges"]
                if item["alert"].get("threat_type")
            })
            attack_states = sorted({
                str(item["alert"].get("attack_state"))
                for item in edge["alert_edges"]
                if item["alert"].get("attack_state")
            })
            status_codes = [
                self._extract_status_code(item["alert"].get("response_headers"))
                for item in edge["alert_edges"]
            ]
            status_codes = [code for code in status_codes if code]
            parts = [
                f"攻击流观察[{self._endpoint_label(edge.get('src'))}→{self._endpoint_label(edge.get('dst'))}]",
                f"告警数={len(edge['alert_edges'])}",
            ]
            if threat_types:
                parts.append(f"来源威胁类型={','.join(threat_types)}")
            if attack_states:
                parts.append(f"来源状态={','.join(attack_states)}")
            threat_types = sorted({
                str(item["alert"].get("threat_type"))
                for item in edge["alert_edges"]
                if item["alert"].get("threat_type")
            })
            attack_states = sorted({
                str(item["alert"].get("attack_state"))
                for item in edge["alert_edges"]
                if item["alert"].get("attack_state")
            })
            status_codes = [
                self._extract_status_code(item["alert"].get("response_headers"))
                for item in edge["alert_edges"]
            ]
            status_codes = [code for code in status_codes if code]
            parts = [
                f"攻击流观察[{self._endpoint_label(edge.get('src'))}→{self._endpoint_label(edge.get('dst'))}]",
                f"告警数={len(edge['alert_edges'])}",
            ]
            if threat_types:
                parts.append(f"来源威胁类型={','.join(threat_types)}")
            if attack_states:
                parts.append(f"来源状态={','.join(attack_states)}")
            if status_codes:
                parts.append(f"观察到响应码={dict(collections.Counter(status_codes))}")
            facts.append("；".join(parts) + "。")

            for alert_edge in edge["alert_edges"][:max_facts_per_edge]:
                alert = alert_edge["alert"]
                response_code = self._extract_status_code(alert.get("response_headers"))
                request_present = bool(alert.get("request_headers") or alert.get("request_body"))
                response_present = bool(alert.get("response_headers") or alert.get("response_body"))
                parts.append(f"观察到响应码={dict(collections.Counter(status_codes))}")
            facts.append("；".join(parts) + "。")

            for alert_edge in edge["alert_edges"][:max_facts_per_edge]:
                alert = alert_edge["alert"]
                response_code = self._extract_status_code(alert.get("response_headers"))
                request_present = bool(alert.get("request_headers") or alert.get("request_body"))
                response_present = bool(alert.get("response_headers") or alert.get("response_body"))
                facts.append(
                    "关键交互观察[{}]：名称={}；请求上下文={}；响应上下文={}；响应码={}。".format(
                        alert_edge.get("ts", "未知时间"),
                        alert.get("alert_name", "未命名告警"),
                        "存在" if request_present else "缺失",
                        "存在" if response_present else "缺失",
                        response_code if response_code else "未识别",
                    )
                )
                    "关键交互观察[{}]：名称={}；请求上下文={}；响应上下文={}；响应码={}。".format(
                        alert_edge.get("ts", "未知时间"),
                        alert.get("alert_name", "未命名告警"),
                        "存在" if request_present else "缺失",
                        "存在" if response_present else "缺失",
                        response_code if response_code else "未识别",
                    )
                )
        return facts

    def _victim_values(self) -> List[str]:
        """从输入节点动态取得受害资产，不引入样例固定 IP。"""
        return [
            self._vertex_value(vertex_id, vertex)
            for vertex_id, vertex in self.vertices.items()
            if vertex.get("role") == "victim"
        ]

    def _victim_values(self) -> List[str]:
        """从输入节点动态取得受害资产，不引入样例固定 IP。"""
        return [
            self._vertex_value(vertex_id, vertex)
            for vertex_id, vertex in self.vertices.items()
            if vertex.get("role") == "victim"
        ]

    def identify_information_gaps(self) -> List[str]:
        """基于可用数据列出待验证项，避免把观察误写为结论。"""
        gaps = ["缺少 EDR 侧进程创建、网络连接和文件落地证据。"]
        victim_values = self._victim_values()
        if victim_values:
            gaps.append(f"缺少受害资产 {', '.join(victim_values[:3])} 的 Sysmon EventID 1/3/11 记录。")
        else:
            gaps.append("缺少可关联受害资产的 Sysmon EventID 1/3/11 记录。")
        if not self.edges:
            gaps.append("缺少可用网络攻击流，需确认 NDR 图数据是否完整。")
        if len(victim_values) <= 1:
            gaps.append("仅观察到单一或未识别受害资产，需验证是否存在横向移动。")
        gaps.append("缺少可确认持久化、C2 通信或命令执行结果的端点证据。")
        return gaps

    def _build_raw_summary(self) -> str:
        """构造保守摘要，供后续 LLM 读取而不丢失结构诊断。"""
        lines = ["=== NDR 安全事件图（观察摘要） ==="]
        shown_alerts = 0
        total_alerts = sum(len(edge.get("alert_edges", [])) for edge in self.edges)
        for edge in self.edges:
            lines.append(
                "流: {} -> {} | 告警数:{}".format(
                    self._endpoint_label(edge.get("src")),
                    self._endpoint_label(edge.get("dst")),
                    len(edge.get("alert_edges", [])),
                )
            )
            for alert_edge in edge.get("alert_edges", [])[:3]:
                shown_alerts += 1
                alert = alert_edge.get("alert", {})
                lines.append(
                    "  - [{}] {} (source_state:{})".format(
                        alert_edge.get("ts", "未知时间"),
                        alert.get("alert_name", "未命名告警"),
                        alert.get("attack_state", "unknown"),
                    )
                )
        lines.append(f"展示代表告警 {shown_alerts}/{total_alerts} 条；完整证据见 evidence_records。")
        if self.diagnostics:
            lines.append("结构诊断: " + " | ".join(self.diagnostics))
        return "\n".join(lines)

    def _infer_semantics(self) -> AlertSemantics:
        """保持保守语义：来源标签只是候选，不能替代 LLM 研判。"""
        threat_types = sorted({
            str(item["alert"].get("threat_type"))
            for edge in self.edges
            for item in edge.get("alert_edges", [])
            if item["alert"].get("threat_type")
        })
        return AlertSemantics(
            category="unknown",
            severity="info",
            intent_tags=threat_types[:5],
        )

    def _event_timestamp(self):
        """优先使用合法扩散时间，无法解析时保留当前时间。"""
        value = self.data.get("diffused_at")
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                self.diagnostics.append("diffused_at 不是可解析时间，已使用当前时间。")
        return datetime.now()

            for item in edge.get("alert_edges", [])
            if item["alert"].get("threat_type")
        })
        return AlertSemantics(
            category="unknown",
            severity="info",
            intent_tags=threat_types[:5],
        )

    def _event_timestamp(self):
        """优先使用合法扩散时间，无法解析时保留当前时间。"""
        value = self.data.get("diffused_at")
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                self.diagnostics.append("diffused_at 不是可解析时间，已使用当前时间。")
        return datetime.now()

    def to_structured_alert(self) -> StructuredAlert:
        """输出可供后续 Agent 调查的保守结构化告警。"""
        tenant = str(self.data.get("tenant") or "UNKNOWN")
        note_parts = [
            f"NDR事件图：{len(self.vertices)} 节点，{len(self.edges)} 边，{len(self.evidences)} 证据。"
        ]
        if self.diagnostics:
            note_parts.append("结构诊断：" + " | ".join(self.diagnostics))
        """输出可供后续 Agent 调查的保守结构化告警。"""
        tenant = str(self.data.get("tenant") or "UNKNOWN")
        note_parts = [
            f"NDR事件图：{len(self.vertices)} 节点，{len(self.edges)} 边，{len(self.evidences)} 证据。"
        ]
        if self.diagnostics:
            note_parts.append("结构诊断：" + " | ".join(self.diagnostics))
        return StructuredAlert(
            alert_id=f"NDR-{tenant}-{self.data.get('diffused_at', '')}",
            raw_alert=self._build_raw_summary(),
            timestamp=self._event_timestamp(),
            alert_id=f"NDR-{tenant}-{self.data.get('diffused_at', '')}",
            raw_alert=self._build_raw_summary(),
            timestamp=self._event_timestamp(),
            source_system="NDR",
            entities=self.extract_entities(),
            semantics=self._infer_semantics(),
            atomic_facts=self.generate_atomic_facts(),
            information_gaps=self.identify_information_gaps(),
            unstructured_notes=" ".join(note_parts),
            raw_payload=self.bundle.raw_payload,
            normalized_payload=self.bundle.normalized_payload,
            json_profile=self.bundle.profile_as_dicts(),
            evidence_records=self.bundle.evidence_as_dicts(),
            input_diagnostics=self.diagnostics,
        )
