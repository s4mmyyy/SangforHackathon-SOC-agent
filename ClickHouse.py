from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from alert_intent_parser import AlertEntity


class ClickHouseQueryGenerator:
    """基于当前假设和信息缺口生成 ClickHouse 查询建议。"""

    QUERY_TEMPLATES = {
        "process_creation": """
            SELECT * FROM sysmon_events 
            WHERE EventID = 1 
            AND {time_filter}
            AND (CommandLine ILIKE '%{keyword}%' OR Image ILIKE '%{keyword}%')
            ORDER BY UtcTime DESC
            LIMIT 50
        """,
        "network_connection": """
            SELECT * FROM sysmon_events 
            WHERE EventID = 3 
            AND {time_filter}
            AND (DestinationIp = '{ip}' OR SourceIp = '{ip}')
            ORDER BY UtcTime DESC
        """,
        "file_creation": """
            SELECT * FROM sysmon_events 
            WHERE EventID = 11 
            AND {time_filter}
            AND TargetFilename ILIKE '%{filename}%'
            ORDER BY UtcTime DESC
        """,
        "dns_query": """
            SELECT * FROM sysmon_events 
            WHERE EventID = 22 
            AND {time_filter}
            AND QueryName ILIKE '%{domain}%'
            ORDER BY UtcTime DESC
        """,
    }

    @staticmethod
    def _entity_type_value(entity: object) -> str:
        """兼容枚举和测试替身，避免查询模块依赖 LLM SDK。"""
        entity_type = getattr(entity, "type", "")
        return str(getattr(entity_type, "value", entity_type)).lower()

    def generate_queries(
        self, gaps: List[str], entities: List["AlertEntity"]
    ) -> List[Dict[str, str]]:
        """根据信息缺口和实体生成查询建议。"""
        queries: List[Dict[str, str]] = []
        time_filter = "UtcTime >= now() - INTERVAL 7 DAY"  # 当前阶段保留原有时间窗口。

        for gap in gaps:
            if "进程" in gap or "Process" in gap:
                for entity in entities:
                    if (
                        self._entity_type_value(entity) == "ip"
                        and getattr(entity, "role", "unknown") == "victim"
                    ):
                        queries.append({
                            "purpose": f"查询受害主机 {getattr(entity, 'value', '')} 上的异常进程",
                            "sql": self.QUERY_TEMPLATES["process_creation"].format(
                                time_filter=time_filter, keyword="powershell"
                            ),
                            "rationale": "攻击成功后常伴随PowerShell/命令执行",
                        })

            if "C2" in gap or "外联" in gap:
                domains = [
                    getattr(entity, "value", "")
                    for entity in entities
                    if self._entity_type_value(entity) == "domain"
                ]
                for domain in filter(None, domains):
                    queries.append({
                        "purpose": f"查询DNS解析记录: {domain}",
                        "sql": self.QUERY_TEMPLATES["dns_query"].format(
                            time_filter=time_filter, domain=domain
                        ),
                        "rationale": "DNSLog平台的外联可能确认命令执行成功",
                    })

            if "WebShell" in gap or "文件落地" in gap:
                queries.append({
                    "purpose": "查询可疑Web文件创建",
                    "sql": self.QUERY_TEMPLATES["file_creation"].format(
                        time_filter=time_filter, filename=".jsp"
                    ),
                    "rationale": "WebShell常以jsp/php/asp形式落地",
                })

        return queries
