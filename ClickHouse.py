class ClickHouseQueryGenerator:
    """
    基于当前假设和信息缺口，生成ClickHouse查询建议
    """
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
        """
    }
    
    def generate_queries(self, gaps: List[str], entities: List[AlertEntity]) -> List[Dict]:
        """
        根据信息缺口和实体生成查询建议
        """
        queries = []
        time_filter = "UtcTime >= now() - INTERVAL 7 DAY"  # 动态调整
        
        for gap in gaps:
            # 针对缺失进程证据
            if "进程" in gap or "Process" in gap:
                for e in entities:
                    if e.type == EntityType.IP and e.role == "victim":
                        queries.append({
                            "purpose": f"查询受害主机 {e.value} 上的异常进程",
                            "sql": self.QUERY_TEMPLATES["process_creation"].format(
                                time_filter=time_filter, keyword="powershell"
                            ),
                            "rationale": "攻击成功后常伴随PowerShell/命令执行"
                        })
            
            # 针对C2/外联证据
            if "C2" in gap or "外联" in gap:
                domains = [e.value for e in entities if e.type == EntityType.DOMAIN]
                for domain in domains:
                    queries.append({
                        "purpose": f"查询DNS解析记录: {domain}",
                        "sql": self.QUERY_TEMPLATES["dns_query"].format(
                            time_filter=time_filter, domain=domain
                        ),
                        "rationale": "DNSLog平台的外联可能确认命令执行成功"
                    })
            
            # 针对文件落地
            if "WebShell" in gap or "文件落地" in gap:
                queries.append({
                    "purpose": "查询可疑Web文件创建",
                    "sql": self.QUERY_TEMPLATES["file_creation"].format(
                        time_filter=time_filter, filename=".jsp"
                    ),
                    "rationale": "WebShell常以jsp/php/asp形式落地"
                })
        
        return queries