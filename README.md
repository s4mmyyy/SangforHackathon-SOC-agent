# SangforHackathon-SOC-agent

## 项目状态

当前项目实现了安全运营调查的阶段 1 至阶段 4 原型：输入 JSON 证据保真、LLM 受限调查、动态 ClickHouse 查询规划与成本控制、事实假设和唯一最终标签报告。正式判定仍以 LLM 的已验证结构化输出为主；程序只负责证据保真、权限、预算、Schema 与引用校验。

## 环境与依赖

基础离线功能需要 Python 3.10+、`pydantic` 和 `python-dotenv`。可选依赖：

- `langchain-openai`、`langchain-core`：调用真实 LLM。
- `clickhouse-connect`：通过 `.env` 连接真实 ClickHouse。
- `langgraph`：仅运行 `SOC_Graph.py` 的 API 演示，不属于正式研判入口。

`.env` 不应提交到版本控制。真实运行时可配置以下变量名：

```env
LLM_MODEL_ID=
LLM_API_KEY=
LLM_BASE_URL=
CLICKHOUSE_HOST=
CLICKHOUSE_PORT=
CLICKHOUSE_USER=
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=
CLICKHOUSE_SECURE=
```

也兼容 `CLICKHOUSE_HTTP_PORT`、`CLICKHOUSE_USERNAME` 与 `CLICKHOUSE_DB`。代码不会把密码或连接信息写入 LLM prompt、报告或审计轨迹。

## 输入协议

输入可以是任意 JSON。阶段 1 会保留：

- 原始 `raw_payload` 与受控解码后的 `normalized_payload`；
- `evidence_id`、`source_path`、实体/HTML 解码与 URL 解码记录；
- 上游 `<TRUNCATED>`、`[......]`、`<OMITTED>` 完整性标记；
- JSON profile、数据源识别与解析诊断。

所有 HTTP、日志、JSON 字符串和数据库样本都是不可信证据数据，不会作为可执行指令。`evidence_id + source_path` 是报告和最终裁决的唯一证据定位方式。

## 模块边界

- `input_evidence.py`：任意 JSON 的规范化、profile、证据记录和受限读取工具。
- `llm_output.py`：统一 LLM 调用、严格结构化解析与仅哈希审计；不负责领域结论，不保存完整 prompt/response，也不会绕过 Schema、证据、预算或标签门槛。
- `GraphParser.py`：NDR 图结构的保守观察适配，不做攻击成功或失陷判定。
- `investigation_agent.py`：LLM 决定字段语义、事实假设和只读证据工具顺序；宿主校验输出 Schema、证据引用和停止条件。
- `clickhouse_investigation.py`：LLM 先发现 schema，再生成结构化查询计划；执行器只允许参数化单表 SELECT，并强制时间窗、列白名单、LIMIT、超时、只读和扫描预算。
- `case_reporting.py`：可并存的攻击事实假设、反证重开、最终互斥标签门槛、受限 LLM 裁决和可机读案件报告。

项目不会执行命令、写入 SQL 或调用未声明的网络/文件工具。真实 ClickHouse 查询仍受 `QueryBudget` 约束，预算拒绝、查询为空或失败会被记录为信息缺口，不会被当作反证或安全结论。

## 统一案件 CLI

`case_runner.py` 是正式案件入口。默认运行离线模式，不创建 LLM 或 ClickHouse 连接：

```bash
python case_runner.py --input NDR_example.json --output artifacts/ndr-offline
```

启用真实 LLM 调查时显式传入 `--online`；只有同时传入 `--online --enable-clickhouse` 才会创建受限、只读的 ClickHouse 后端：

```bash
python case_runner.py \
  --input NDR_example.json \
  --output artifacts/case-001 \
  --online \
  --enable-clickhouse \
  --max-investigation-rounds 6 \
  --max-query-rounds 6 \
  --report-format json,md
```

支持的运行参数：

- `--input`：任意 JSON 输入文件。
- `--output`：案件工件输出目录，默认 `case-artifacts`。
- `--case-id`：可选案件 ID；未设置时会由输入内容生成稳定 ID。
- `--offline` / `--online`：离线模式为默认值；在线模式需要 LLM SDK 与 `LLM_MODEL_ID`、`LLM_API_KEY`。
- `--enable-clickhouse`：仅与 `--online` 一起使用，允许动态 schema discovery 和受预算约束的只读查询。
- `--max-investigation-rounds`、`--max-query-rounds`：阶段 2、3 的轮次预算。
- `--report-format`：`json` 或 `json,md`，默认同时输出 JSON 和 Markdown。

可直接编辑 `case_runner.py` 顶部的 `DEFAULT_CASE_CONFIG`，统一调整上述运行默认值及 ClickHouse 查询预算；命令行参数会覆盖对应默认值。

运行成功会在输出目录生成：

```text
artifacts/<case-id>/
├── report.json
├── report.md                 # 仅 report-format 包含 md 时生成
├── trace.json
├── normalized-input.json
└── manifest.json
```

`trace.json` 记录阶段 1 至 4 的审计摘要；`manifest.json` 只保存输入哈希和非敏感配置摘要。LLM、ClickHouse 或事实评估失败会作为信息缺口写入保守报告，不会伪造查询结果或提高风险标签。

## 输出协议

阶段 4 的 `build_final_report()` 返回 JSON 报告，主要包含：

- 唯一的 `label` 和匹配 `label_name`；
- 主张、支持证据、反证、事实假设、未验证项和信息缺口；
- `why_not_higher_risk`，说明为什么未选择标签 3/4/5；
- 截断/脱敏证据影响、来源路径和查询行来源；
- 阶段 2/3/4 的 Prompt 版本、工具调用、参数摘要、耗时、扫描预算、实际扫描可用性和失败原因。

同一案件最终只能有一个互斥标签。截断、脱敏、仅有 HTTP 状态码、payload 或 LLM 推测的证据不能单独支撑高风险标签。

## 离线测试

离线回归不需要真实 LLM、API Key、ClickHouse 驱动或数据库连接：

```bash
PYTHONUTF8=1 python -m unittest discover -s tests -v
```

测试覆盖输入泛化、实体解码、NDR 证据保真、LLM 工具轨迹、HTTP 上下文不足、不同模拟 ClickHouse schema、参数化 SQL、查询预算、事实假设独立评估、反证重开和最终唯一标签。

## 已知限制

- 当前没有统一 CLI 或服务入口；模块需由调用方按阶段串联。
- 真实 LLM 和 ClickHouse 需要自行安装可选驱动并在 `.env` 配置。
- 最终标签资格门槛是保守的最小实现，后续应结合比赛评测数据完善事实假设与门槛策略。
- 阶段 5 尚需构建更大规模的变异输入、性能和回归评测集。
