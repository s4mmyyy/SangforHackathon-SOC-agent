from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    alert: str
    hypotheses: dict[str, float]
    tool_result: str
    conclusion: str

# 节点1：理解告警，初始化假设
def init_node(state: State):
    # 调用 LLM 解析 alert，生成初始假设
    return {"hypotheses": {"误报": 0.4, "扫描": 0.4, "入侵": 0.2}}

# 节点2：调用工具查证据
def tool_node(state: State):
    # 模拟查询日志
    return {"tool_result": "发现该IP连接了已知恶意域名"}

# 节点3：贝叶斯更新（自研逻辑）
def bayesian_update(state: State):
    # 你的自研算法：根据 tool_result 更新 hypotheses
    h = state["hypotheses"].copy()
    h["入侵"] = 0.85  # 证据支持入侵
    return {"hypotheses": h, "conclusion": "高置信度入侵"}

# 构建图
builder = StateGraph(State)
builder.add_node("init", init_node)
builder.add_node("tool", tool_node)
builder.add_node("update", bayesian_update)
builder.set_entry_point("init")
builder.add_edge("init", "tool")
builder.add_edge("tool", "update")
builder.add_edge("update", END)

graph = builder.compile()
result = graph.invoke({"alert": "服务器异常外联告警"})
print(result)