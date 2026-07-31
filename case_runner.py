"""统一案件编排 CLI。默认离线，不读取或连接外部服务。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from case_orchestrator import CaseRunConfig, CaseRunResult, run_case
from clickhouse_investigation import QueryBudget, create_env_clickhouse_backend
from report_renderer import render_markdown_report


# 可按本地演练需要编辑；命令行参数会覆盖这些默认值。
DEFAULT_CASE_CONFIG: Dict[str, Any] = {
    "online": False,
    "clickhouse_enabled": False,
    "stage2_max_rounds": 6,
    "stage3_max_rounds": 6,
    "query_budget": {
        "max_rows_returned": 200,
        "max_rows_scanned": 200_000,
        "max_bytes_scanned": 64 * 1024 * 1024,
        "max_timeout_seconds": 30,
        "max_window_minutes": 24 * 60,
    },
}


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"输入文件不存在: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"输入文件不是 UTF-8 JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"输入文件不是有效 JSON: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行保守、可审计的 SOC 案件编排。默认离线。")
    parser.add_argument("--input", required=True, type=Path, help="输入 JSON 文件")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=Path("case-artifacts"), help="案件工件输出目录")
    parser.add_argument("--case-id", help="覆盖由输入摘要生成的案件 ID")
    parser.add_argument("--source-system", help="输入来源系统标识")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--online", dest="online", action="store_true", help="显式允许创建 LLM 客户端")
    mode.add_argument("--offline", dest="online", action="store_false", help="强制离线模式（默认）")
    parser.set_defaults(online=bool(DEFAULT_CASE_CONFIG["online"]))
    clickhouse = parser.add_mutually_exclusive_group()
    clickhouse.add_argument("--enable-clickhouse", "--clickhouse", dest="clickhouse_enabled", action="store_true", help="在线模式下允许受限 ClickHouse 调查")
    clickhouse.add_argument("--no-clickhouse", dest="clickhouse_enabled", action="store_false", help="禁用 ClickHouse（默认）")
    parser.set_defaults(clickhouse_enabled=bool(DEFAULT_CASE_CONFIG["clickhouse_enabled"]))
    parser.add_argument("--max-investigation-rounds", "--stage2-max-rounds", dest="stage2_max_rounds", type=int, default=int(DEFAULT_CASE_CONFIG["stage2_max_rounds"]))
    parser.add_argument("--max-query-rounds", "--stage3-max-rounds", dest="stage3_max_rounds", type=int, default=int(DEFAULT_CASE_CONFIG["stage3_max_rounds"]))
    parser.add_argument("--report-format", default="json,md", help="输出格式：json 或 json,md")
    return parser


def _parse_report_formats(value: str) -> set[str]:
    formats = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not formats or not formats.issubset({"json", "md"}) or "json" not in formats:
        raise ValueError("--report-format 仅支持 json 或 json,md。")
    return formats


def _validate_args(args: argparse.Namespace) -> set[str]:
    if args.clickhouse_enabled and not args.online:
        raise ValueError("--enable-clickhouse 需要同时显式指定 --online。")
    if args.stage2_max_rounds < 1 or args.stage3_max_rounds < 1:
        raise ValueError("阶段最大轮数必须为正整数。")
    return _parse_report_formats(args.report_format)


def _make_online_llm() -> Any:
    """仅由 --online 调用，复用项目已有适配器且不在工件记录配置。"""
    import os

    from alert_intent_parser import ChatOpenAIAdapter, _create_default_llm

    missing = [name for name in ("LLM_MODEL_ID", "LLM_API_KEY") if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError("在线模式缺少 LLM 配置：" + ", ".join(missing) + "。请配置后重试或使用 --offline。")
    client = _create_default_llm()
    if client is None:
        raise RuntimeError("在线模式需要安装 langchain-openai 和 langchain-core。")
    return ChatOpenAIAdapter(client)


def config_from_args(args: argparse.Namespace) -> CaseRunConfig:
    budget = QueryBudget(**dict(DEFAULT_CASE_CONFIG["query_budget"]))
    return CaseRunConfig(
        case_id=args.case_id,
        source_system=args.source_system,
        online=args.online,
        clickhouse_enabled=args.clickhouse_enabled,
        stage2_max_rounds=args.stage2_max_rounds,
        stage3_max_rounds=args.stage3_max_rounds,
        query_budget=budget,
    )


def write_artifacts(
    output_dir: Path,
    result: CaseRunResult,
    config: CaseRunConfig,
    input_path: Path,
    report_formats: set[str],
) -> Dict[str, Path]:
    """写入所有工件；manifest 只包含哈希与公开配置摘要。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "normalized_input": output_dir / "normalized-input.json",
        "report": output_dir / "report.json",
        "trace": output_dir / "trace.json",
        "manifest": output_dir / "manifest.json",
    }
    if "md" in report_formats:
        paths["markdown"] = output_dir / "report.md"
    _write_json(paths["normalized_input"], result.normalized_input)
    _write_json(paths["report"], result.report)
    _write_json(paths["trace"], result.trace)
    if "markdown" in paths:
        paths["markdown"].write_text(render_markdown_report(result.report), encoding="utf-8")
    manifest = {
        "case_id": result.report.get("case_id"),
        "status": "completed",
        "input": {"path": str(input_path), "sha256": _hash_bytes(input_path.read_bytes())},
        "config": config.public_dict(),
        "report_formats": sorted(report_formats),
        "artifacts": {
            name: {"path": path.name, "sha256": _hash_bytes(path.read_bytes())}
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    _write_json(paths["manifest"], manifest)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report_formats = _validate_args(args)
        payload = _load_json(args.input)
        config = config_from_args(args)
        llm_client = _make_online_llm() if config.online else None
        startup_gaps = []
        clickhouse_backend = None
        if config.clickhouse_enabled:
            try:
                clickhouse_backend = create_env_clickhouse_backend()
            except RuntimeError:
                startup_gaps.append("ClickHouse 配置不可用，未执行数据库查询。")
        config.startup_gaps = startup_gaps
        result = run_case(payload, config, llm_client=llm_client, clickhouse_backend=clickhouse_backend)
        paths = write_artifacts(args.output, result, config, args.input, report_formats)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2
    print(f"案件工件已写入: {paths['manifest'].parent}")
    print(f"最终标签: {result.report['final_adjudication']['label_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
