# SangforHackathon-SOC-agent

## 离线稳定性测试

当前阶段的回归测试不需要真实 LLM、API Key 或网络连接，执行：

```bash
python -m unittest discover -s tests -v
```

测试覆盖 ClickHouse 模块导入、假设管理器的离线降级、NDR JSON 异常输入诊断、`NDR_example.json` smoke test，以及 LangGraph 演示模块的安全导入。

`SOC_Graph.py` 仅用于 LangGraph API 演示，不是正式安全研判入口；实际运行演示需要另行安装 `langgraph`。