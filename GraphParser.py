from __future__ import annotations   # 若类型注解需要使用字符串，可选
import json
import re
from datetime import datetime
from typing import List
import collections
from alert_intent_parser import StructuredAlert, AlertSemantics, AlertEntity, EntityType

# ===== 新增：NDR 图解析器 =====
class NDRGraphParser:
    """
    将赛题的 NDR JSON 安全事件图解析为意图理解模块可消费的格式
    """
    def __init__(self, ndr_json: dict, llm_client=None):
        self.data = ndr_json
        self.llm = llm_client
        self.response_analyzer = LLMResponseAnalyzer(llm_client) if llm_client else None
        self.vertices = {v["id"]: v for v in ndr_json.get("vertices", [])}
        self.edges = ndr_json.get("main_edges", [])
        self.evidences = ndr_json.get("evidences", [])
    
    def extract_entities(self) -> List[AlertEntity]:
        """从 vertices 提取实体"""
        entities = []
        for vid, v in self.vertices.items():
            entity_type = self._map_vertex_type(v["type"])
            role = v.get("role", "unknown")
            # 从 properties 提取 IP
            value = v.get("properties", {}).get("ip", vid.split(":")[-1])
            entities.append(AlertEntity(
                value=value,
                type=entity_type,
                role="attacker" if role == "attacker" else "victim" if role == "victim" else "unknown",
                confidence=1.0,
                context=json.dumps(v.get("properties", {}), ensure_ascii=False)
            ))
        return entities
    
    def _map_vertex_type(self, t: str) -> EntityType:
        mapping = {
            "IP": EntityType.IP, "DOMAIN": EntityType.DOMAIN,
            "FILE": EntityType.FILE, "PROCESS": EntityType.PROCESS,
            "USER": EntityType.USER
        }
        return mapping.get(t, EntityType.HASH)
    
    def generate_atomic_facts(self, max_facts_per_edge=3) -> List[str]:
        facts = []
        
        # === 第一层：全局拓扑摘要（1-2条）===
        attackers = [v for v in self.vertices.values() if v.get("role") == "attacker"]
        victims = [v for v in self.vertices.values() if v.get("role") == "victim"]
        facts.append(f"全局拓扑: {len(attackers)}个攻击者({','.join(a['properties']['ip'] for a in attackers)}) "
                    f"→ {len(victims)}个受害者({','.join(v['properties']['ip'] for v in victims)}), "
                    f"共{len(self.edges)}条攻击流, 时间跨度{self.edges[0]['first_seen'][:10]}至{self.edges[-1]['last_seen'][:10]}")
        
        # === 第二层：按攻击流聚合（而非按单条告警展开）===
        for edge in self.edges:
            src_ip = edge["src"].split(":")[-1]
            dst_ip = edge["dst"].split(":")[-1]
            
            # 提取该流的所有攻击类型
            threat_types = set()
            attack_states = set()
            status_codes = []
            has_cmd_payload = False
            has_2xx = False
            has_4xx = False
            has_redirect_to_anonym = False
            
            for ae in edge.get("alert_edges", []):
                alert = ae.get("alert", {})
                threat_types.add(alert.get("threat_type", "unknown"))
                attack_states.add(alert.get("attack_state", "unknown"))
                
                # 关键：只保留有HTTP详情的告警做深度分析
                resp_headers = alert.get("response_headers", "")
                if resp_headers:
                    code = self._extract_status_code(resp_headers)
                    status_codes.append(code)
                    if code == 200: has_2xx = True
                    if code in [400, 403, 404]: has_4xx = True
                    if "anonym.jsp" in resp_headers: has_redirect_to_anonym = True
                
                body = str(alert.get("request_body", ""))
                if any(k in body for k in ["curl", "whoami", "cat ", "/etc/passwd"]):
                    has_cmd_payload = True
            
            # 生成该攻击流的"决策级"事实，而非罗列每条告警
            fact_parts = [f"攻击流[{src_ip}→{dst_ip}]: 类型={','.join(threat_types)}, 状态={','.join(attack_states)}"]
            
            if status_codes:
                fact_parts.append(f"响应模式={collections.Counter(status_codes).most_common()}")
            if has_redirect_to_anonym:
                fact_parts.append("存在WAF拦截特征(302→anonym.jsp)")
            if has_cmd_payload:
                fact_parts.append("包含命令执行类payload")
            if has_2xx and not has_4xx:
                fact_parts.append("全部200响应，无拦截迹象")
            elif has_4xx and not has_2xx:
                fact_parts.append("全部4xx响应，疑似被拦截")
            
            facts.append(" | ".join(fact_parts))
            
            # === 第三层：只保留"高价值"单条告警（有完整HTTP交互的）===
            high_value_alerts = [
                ae for ae in edge.get("alert_edges", [])
                if ae.get("alert", {}).get("response_headers")  # 有响应头才值得LLM分析
            ]
            # 每流最多保留3条典型告警
            for ae in high_value_alerts[:max_facts_per_edge]:
                alert = ae["alert"]
                facts.append(
                    f"关键交互[{ae.get('ts', '')}]: {alert.get('alert_name')} | "
                    f"状态码={self._extract_status_code(alert.get('response_headers', ''))} | "
                    f"payload特征={self._summarize_payload(alert.get('request_body', ''))} | "
                    f"响应语义={self._summarize_response(alert.get('response_body', ''))}"
                )
        
        return facts

    def _summarize_payload(self, body: str) -> str:
        """提取payload关键特征，而非全文"""
        if not body: return "无"
        if "curl" in body or "wget" in body: return "命令执行/curl外联"
        if "alert(document.domain)" in body: return "XSS探测"
        if "SLEEP" in body.upper(): return "SQLi时延探测"
        if "ldap://" in body.lower(): return "LDAP注入"
        if len(body) > 200: return body[:100] + "..."
        return body

    def _summarize_response(self, body: str) -> str:
        """提取响应关键语义"""
        if not body: return "空"
        if "400" in body or "错误的请求" in body: return "400错误页"
        if "anonym.jsp" in body: return "WAF拦截页"
        if len(body) > 200: return body[:100] + "..."
        return body
    
    def identify_information_gaps(self) -> List[str]:
        """
        基于 NDR 图识别信息缺口，驱动后续 EDR 调查
        """
        gaps = []
        
        # 检查是否有 EDR 侧数据
        gaps.append("缺少 EDR 侧进程创建/网络连接/文件落地证据")
        gaps.append("缺少目标主机 10.10.10.112 上的 Sysmon EventID 1/3/11 记录")
        
        # 检查响应确认
        has_success_response = any(
            self._extract_status_code(a.get("alert", {}).get("response_headers", "")) in [200, 201]
            for edge in self.edges
            for a in edge.get("alert_edges", [])
        )
        if not has_success_response:
            gaps.append("所有告警响应均为非200状态，需确认WAF/IPS拦截详情")
        
        # 检查是否有命令执行成功证据
        has_cmd_exec_confirm = any(
            "成功" in str(a.get("alert", {}).get("response_body", "")) or 
            "root:" in str(a.get("alert", {}).get("response_body", ""))
            for edge in self.edges
            for a in edge.get("alert_edges", [])
        )
        if not has_cmd_exec_confirm:
            gaps.append("未发现命令执行成功回显（如/etc/passwd内容、whoami结果）")
        
        # 检查横向移动
        victim_ips = [v.get("properties", {}).get("ip") for v in self.vertices.values() if v.get("role") == "victim"]
        if len(victim_ips) <= 1:
            gaps.append("仅发现单一受害IP，需排查是否存在横向移动")
        
        # 检查持久化证据
        gaps.append("缺少 WebShell 文件落地/注册表修改/计划任务等持久化证据")
        gaps.append("缺少 C2 通信/隧道建立的网络连接证据")
        
        return gaps
    
    def _extract_status_code(self, headers: str) -> int:
        """从响应头提取状态码"""
        if not headers:
            return 0
        match = re.search(r'HTTP/\d\.\d\s+(\d+)', headers)
        return int(match.group(1)) if match else 0
    
    def to_structured_alert(self) -> StructuredAlert:
        """输出统一格式的 StructuredAlert"""
        # 构造原始文本摘要（供LLM理解）
        raw_summary = self._build_raw_summary()
        
        return StructuredAlert(
            alert_id=f"NDR-{self.data.get('tenant', 'UNKNOWN')}-{self.data.get('diffused_at', '')}",
            raw_alert=raw_summary,
            timestamp=datetime.now(),  # 或用 diffused_at
            source_system="NDR",
            entities=self.extract_entities(),
            semantics=self._infer_semantics(),
            atomic_facts=self.generate_atomic_facts(),
            information_gaps=self.identify_information_gaps(),
            unstructured_notes=f"NDR事件图: {len(self.vertices)} 节点, {len(self.edges)} 边, {len(self.evidences)} 证据"
        )
    
    def _build_raw_summary(self) -> str:
        """将JSON图压缩为文本，供LLM快速理解"""
        lines = ["=== NDR 安全事件图 ==="]
        for edge in self.edges:
            src = edge["src"].split(":")[-1]
            dst = edge["dst"].split(":")[-1]
            lines.append(f"攻击流: {src} -> {dst} | {edge.get('occurrence_pattern')} | 告警数:{edge.get('alert_count')}")
            for ae in edge.get("alert_edges", [])[:3]:  # 只取前3条避免过长
                alert = ae.get("alert", {})
                lines.append(f"  - [{ae.get('ts')}] {alert.get('alert_name')} (state:{alert.get('attack_state')})")
        return "\n".join(lines)
    
    def _infer_semantics(self) -> AlertSemantics:
        """从全局图推断语义（而非单条告警）"""
        # 收集所有攻击类型
        all_threats = set()
        all_states = set()
        for edge in self.edges:
            for ae in edge.get("alert_edges", []):
                alert = ae.get("alert", {})
                all_threats.add(alert.get("threat_type", ""))
                all_states.add(alert.get("attack_state", ""))
        
        # 判断语义类别
        category = "unknown"
        if any("代码注入" in t for t in all_threats):
            category = "intrusion"
        elif any("SQL注入" in t for t in all_threats):
            category = "intrusion"
        elif any("XSS" in t for t in all_threats):
            category = "intrusion"
        elif any("目录遍历" in t for t in all_threats):
            category = "reconnaissance"
        
        # 判断严重级别
        severity = "medium"
        if "success" in all_states:
            severity = "critical"
        elif any("命令注入" in t for t in all_threats):
            severity = "high"
        
        return AlertSemantics(
            category=category,
            tactic="多向量Web攻击（扫描+注入+遍历）",
            severity=severity,
            intent_tags=list(all_threats)[:5]
        )