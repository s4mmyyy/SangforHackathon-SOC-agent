import json
from collections import Counter

# 加载你的example.json
with open("example.json", encoding='utf-8') as f:
    data = json.load(f)

# 模拟精简逻辑
facts = []
for edge in data["main_edges"]:
    src = edge["src"].split(":")[-1]
    dst = edge["dst"].split(":")[-1]
    
    # 只统计有HTTP详情的alert
    detailed_alerts = [
        ae for ae in edge.get("alert_edges", [])
        if ae.get("alert", {}).get("response_headers")
    ]
    
    threat_types = set(ae["alert"].get("threat_type", "unknown") 
                      for ae in edge.get("alert_edges", []))
    states = set(ae["alert"].get("attack_state", "unknown") 
                for ae in edge.get("alert_edges", []))
    
    facts.append(f"流[{src}→{dst}]: 类型={threat_types}, 状态={states}, 详细告警数={len(detailed_alerts)}")

print(f"原始原子事实: ~288条")
print(f"精简后决策事实: {len(facts)}条")
for f in facts:
    print(f"  {f}")