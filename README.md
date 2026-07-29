# SangforHackathon-SOC-agent

## 数据流

NDR 图数据先由 `GraphParser.NDRGraphParser` 归一化为 `CaseContext` 与不可变风格的 `EvidenceRecord` 证据账本；`HypothesisManager` 调用 `LabelPolicy`，仅依据显式结果和经白名单校验的结构化事实给出 1–5 标签。需要补证时，调查缺口可转换为有边界的查询任务；查询结果默认作为 `UNKNOWN` 观测保留，只有已映射的安全控制动作字段给出明确 `blocked/denied` 等值时才会记录阻断，不会因 HTTP 状态码等原始字段自动推断攻击结果。

## 模块职责与模式

- `evidence_models.py`：案件、实体、证据、调查任务、查询结果与审计契约。
- `GraphParser.py`：NDR 图解析和旧告警兼容视图，不调用 LLM。
- `label_policy.py`：确定性标签策略。
- `hypothesis_manager.py`：证据账本、调查任务与旧假设接口兼容层。
- `schema_discovery.py`：从注入的查询客户端发现通用字段语义，不硬编码客户 Schema。
- `clickhouse_adapter.py`：使用发现结果构造受限的只读查询。
- `SOC_Graph.py`：编排 NDR-only 与可选跨源调查流程。

支持三种使用模式：

- **NDR**：仅解析 NDR 并完成离线研判。
- **EDR**：以已归一化的端点证据账本参与策略研判。
- **N+E**：在已验证 NDR 与端点/查询数据源的租户、资产关联后，补充跨源证据。

## 安全查询与可选依赖

ClickHouse 客户端必须由调用方注入；项目不会自行连接数据库。适配器只生成经过标识符校验的 `SELECT`，强制时间范围、行数上限、参数化实体过滤及只读查询设置，并在映射不适合或有歧义时拒绝查询。`langchain`、`langgraph` 和 ClickHouse 客户端包均为可选依赖；缺失时 NDR-only 路径仍可运行。本项目不声明已建立真实 ClickHouse 连通性。

## 测试

离线标准库测试使用内存 fake client，不需要网络、`.env` 或可选依赖：

```bash
python -m unittest discover -s tests -v
```
