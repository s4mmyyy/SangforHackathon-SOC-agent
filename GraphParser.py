from typing import List, Optional, Literal

# ===== 新增：NDR 图解析器 =====
class NDRGraphParser:
    def __init__(self, ndr_json: dict, llm_client=None):
        self.data = ndr_json
        self.llm = llm_client
        self.response_analyzer = LLMResponseAnalyzer(llm_client) if llm_client else None
        # ...
    
    def generate_atomic_facts(self) -> List[str]:
        facts = []
        
        for edge in self.edges:
            # ... 原有逻辑 ...
            
            for alert_edge in edge.get("alert_edges", []):
                alert = alert_edge.get("alert", {})
                
                # ===== 核心LLM介入点：分析每条告警的HTTP交互 =====
                if self.response_analyzer and alert.get("request_headers"):
                    analysis = self.response_analyzer.analyze(
                        request_headers=alert.get("request_headers", ""),
                        request_body=alert.get("request_body", ""),
                        response_headers=alert.get("response_headers", ""),
                        response_body=alert.get("response_body", ""),
                        alert_name=alert.get("alert_name", "")
                    )
                    
                    result = analysis.get("result", "suspicious")
                    confidence = analysis.get("confidence", 0.5)
                    reasoning = analysis.get("reasoning", "")
                    
                    # 将LLM分析结果转化为原子事实
                    facts.append(f"🤖 LLM分析[{alert.get('alert_name')}]: 攻击结果={result}, 置信度={confidence:.2f}")
                    facts.append(f"🤖 LLM推理: {reasoning[:150]}")
                    
                    # 根据LLM判断生成高价值原子事实
                    if result == "blocked":
                        facts.append(f"✅ 攻击被拦截: {alert.get('alert_name')} (LLM置信度{confidence:.0%})")
                    elif result == "success":
                        facts.append(f"🚨 攻击可能成功: {alert.get('alert_name')} (LLM置信度{confidence:.0%})")
                        facts.append(f"🚨 成功指标: {', '.join(analysis.get('indicators', []))}")
                    elif result == "failed":
                        facts.append(f"❌ 攻击执行失败: {alert.get('alert_name')} (目标不存在或语法错误)")
                    
                    # 覆盖简单的状态码规则
                    facts.append(f"📊 响应语义: {analysis.get('response_meaning', '')}")
                
                # ... 原有逻辑 ...
        
        return facts