"""无副作用的 SOC 案件编排层。

NDR 输入先经 GraphParser 的旧兼容视图（可用时），随后转换为稳定的结构化
证据账本。最终标签完全由 HypothesisManager 内的 LabelPolicy 决定。
"""

from __future__ import annotations

from typing import Any, TypedDict

from clickhouse_adapter import ClickHouseAdapter
from evidence_models import AuditTrace, CaseContext, EvidenceRecord, LabelDecision, QueryResult
from hypothesis_manager import HypothesisManager
from schema_discovery import QueryClientProtocol, SchemaDiscovery


class CaseState(TypedDict, total=False):
    raw_ndr: dict[str, Any]
    legacy_alert: Any
    case_context: CaseContext
    evidence_ledger: list[EvidenceRecord]
    decision: LabelDecision
    tasks: list[Any]
    query_results: list[QueryResult]
    audit: list[AuditTrace]
    remaining_query_budget: int
    stop_reason: str


class _DirectWorkflow:
    """没有 LangGraph 时提供同形 ``invoke`` 的直接回退。"""

    def __init__(
        self,
        client: QueryClientProtocol | None,
        query_budget: int,
        verified_cross_source_context: bool,
    ) -> None:
        self.client = client
        self.query_budget = query_budget
        self.verified_cross_source_context = verified_cross_source_context

    def invoke(self, state: dict[str, Any]) -> CaseState:
        raw_ndr = state.get("raw_ndr", state.get("ndr_data", state))
        return run_case(
            raw_ndr,
            client=self.client,
            query_budget=self.query_budget,
            verified_cross_source_context=self.verified_cross_source_context,
        )


def _parse_ndr(ndr_data: dict[str, Any]) -> tuple[CaseContext, list[EvidenceRecord], Any]:
    """通过唯一的 NDR 适配器生成案件、结构化证据和 legacy 兼容视图。"""
    if not isinstance(ndr_data, dict):
        raise TypeError("NDR 输入必须是 dict")
    from GraphParser import NDRGraphParser

    parser = NDRGraphParser(ndr_data)
    return parser.to_case_context(), parser.to_evidence_records(), parser.to_structured_alert()


def _query_cycle(
    manager: HypothesisManager,
    client: QueryClientProtocol,
    budget: int,
    *,
    verified_cross_source_context: bool,
) -> tuple[list[QueryResult], list[AuditTrace], int, str | None]:
    if not verified_cross_source_context:
        return [], [], budget, "未验证 NDR 与查询数据源的租户/资产关联，未自动跨源查询。"
    tasks = manager.create_investigation_tasks()
    if not tasks or budget <= 0:
        return [], [], budget, "没有可执行的有界调查任务或查询预算已耗尽。"
    try:
        discovery = SchemaDiscovery(client)
        mappings = discovery.discover_mappings(discovery.discover_catalog())
        adapter = ClickHouseAdapter(client, mappings)
    except Exception as exc:
        trace = AuditTrace(event="schema_discovery_error", success=False, details={"error_type": type(exc).__name__})
        return [], [trace], budget, "Schema 发现失败，保留 NDR-only 结论。"

    results: list[QueryResult] = []
    for task in tasks[:budget]:
        result = adapter.execute(task)
        results.append(result)
        manager.add_structured_evidence(result.evidence)
        # 当前适配器明确把行标为 UNKNOWN；没有可判定新事实时不扩展查询循环。
        if not result.evidence:
            break
    audit = [trace for result in results for trace in result.audit]
    return results, audit, max(0, budget - len(results)), "查询完成；结果仅作为 UNKNOWN 观测保留，未自动提升标签。"


def run_case(
    ndr_data: dict[str, Any],
    *,
    client: QueryClientProtocol | None = None,
    query_budget: int = 0,
    manager: HypothesisManager | None = None,
    verified_cross_source_context: bool = False,
) -> CaseState:
    """执行一个案件；无 LangGraph、无 ClickHouse 时 NDR-only 仍可完成。"""
    if query_budget < 0:
        raise ValueError("query_budget 不能为负数")
    case, records, legacy_alert = _parse_ndr(ndr_data)
    manager = manager or HypothesisManager()
    manager.initialize_from_evidence(case, records)
    state: CaseState = {
        "raw_ndr": ndr_data,
        "case_context": case,
        "evidence_ledger": manager.evidence_ledger,
        "decision": manager.get_label_decision(),
        "tasks": manager.create_investigation_tasks(),
        "query_results": [],
        "audit": [],
        "remaining_query_budget": query_budget,
        "stop_reason": "NDR-only 研判完成。",
    }
    if legacy_alert is not None:
        state["legacy_alert"] = legacy_alert
    if client is not None and query_budget > 0:
        results, audit, remaining, reason = _query_cycle(
            manager, client, query_budget, verified_cross_source_context=verified_cross_source_context
        )
        state.update({
            "evidence_ledger": manager.evidence_ledger,
            "decision": manager.get_label_decision(),
            "tasks": manager.create_investigation_tasks(),
            "query_results": results,
            "audit": audit,
            "remaining_query_budget": remaining,
            "stop_reason": reason or state["stop_reason"],
        })
    return state


def build_workflow(
    client: QueryClientProtocol | None = None,
    *,
    query_budget: int = 0,
    verified_cross_source_context: bool = False,
) -> Any:
    """构建可调用工作流；缺少 LangGraph 时返回直接回退对象。"""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _DirectWorkflow(client, query_budget, verified_cross_source_context)

    def execute_node(state: CaseState) -> CaseState:
        return run_case(
            state.get("raw_ndr", {}),
            client=client,
            query_budget=query_budget,
            verified_cross_source_context=verified_cross_source_context,
        )

    builder = StateGraph(CaseState)
    builder.add_node("run_case", execute_node)
    builder.set_entry_point("run_case")
    builder.add_edge("run_case", END)
    return builder.compile()


if __name__ == "__main__":
    print("SOC_Graph 已加载；请调用 run_case(ndr_data) 或 build_workflow()。")
