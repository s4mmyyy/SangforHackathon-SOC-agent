"""LangGraph 编排演示：仅用于教学，不是正式安全研判入口。"""

from typing import Any, Dict, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # 允许离线测试安全导入本演示文件。
    END = None
    StateGraph = None


class State(TypedDict, total=False):
    """演示图状态：不包含任何真实攻击裁决。"""

    alert: str
    hypotheses: Dict[str, float]
    tool_result: str
    conclusion: str


def init_node(state: State) -> Dict[str, Any]:
    """初始化演示状态，不将样例值当作真实研判概率。"""
    return {
        "hypotheses": {},
        "conclusion": "演示图未接入真实告警解析与裁决器，不能生成研判结论。",
    }


def tool_node(state: State) -> Dict[str, str]:
    """模拟工具节点，仅说明未执行真实日志查询。"""
    return {"tool_result": "演示模式：未执行真实日志或威胁情报查询。"}


def bayesian_update(state: State) -> Dict[str, Any]:
    """演示更新节点：保留输入状态，不生成固定攻击结论。"""
    return {
        "hypotheses": dict(state.get("hypotheses", {})),
        "conclusion": "演示模式：缺少真实证据，需由正式 Agent 完成研判。",
    }


def build_demo_graph():
    """构建无固定结论的 LangGraph API 演示图。"""
    if StateGraph is None:
        raise RuntimeError("运行 SOC_Graph 演示需要安装 langgraph。")

    builder = StateGraph(State)
    builder.add_node("init", init_node)
    builder.add_node("tool", tool_node)
    builder.add_node("update", bayesian_update)
    builder.set_entry_point("init")
    builder.add_edge("init", "tool")
    builder.add_edge("tool", "update")
    builder.add_edge("update", END)
    return builder.compile()


if __name__ == "__main__":
    # 仅直接执行时运行示例，导入模块不会产生副作用。
    graph = build_demo_graph()
    result = graph.invoke({"alert": "演示告警"})
    print(result)
