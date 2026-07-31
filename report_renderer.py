"""将可机读案件报告渲染为简洁、无敏感原文的 Markdown。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _text(value: Any) -> str:
    """单行展示未受信任描述，避免 Markdown 注入和超长输出。"""
    return str(value or "").replace("\r", " ").replace("\n", " ").replace("|", "\\|")[:600]


def _bullets(items: Iterable[Any], empty: str = "无") -> List[str]:
    rendered = [f"- {_text(item)}" for item in items if _text(item)]
    return rendered or [f"- {empty}"]


def _references(items: Iterable[Dict[str, Any]], empty: str = "无可验证引用") -> List[str]:
    rendered = []
    for item in items:
        evidence_id = item.get("evidence_id")
        source_path = item.get("source_path")
        if evidence_id and source_path:
            rendered.append(f"- `{_text(evidence_id)}`: `{_text(source_path)}`")
    return rendered or [f"- {empty}"]


def render_markdown_report(report: Dict[str, Any]) -> str:
    """从 `build_final_report()` 的输出生成适合人工审阅的简洁 Markdown。"""
    final = report.get("final_adjudication", {})
    claim = report.get("primary_claim", {})
    lines = [
        "# 案件报告",
        "",
        f"- 案件 ID: `{_text(report.get('case_id'))}`",
        f"- 最终标签: `{_text(final.get('label'))}` `{_text(final.get('label_name'))}`",
        f"- 置信度: `{_text(final.get('confidence'))}`",
        f"- 裁决模式: `{_text(final.get('decision_mode'))}`",
        "",
        "## 主张",
        "",
        _text(claim.get("claim")) or "未形成可验证主张。",
        "",
        "## 支持证据",
        "",
        *_references(claim.get("supporting_evidence", [])),
        "",
        "## 反证",
        "",
        *_references(report.get("contradicting_evidence", []), "无已记录反证"),
        "",
        "## 事实假设",
        "",
    ]
    hypotheses = report.get("fact_hypotheses", [])
    if hypotheses:
        for item in hypotheses:
            lines.extend([
                f"### `{_text(item.get('kind'))}`: `{_text(item.get('status'))}`",
                "",
                _text(item.get("statement")),
                "",
            ])
            assessments = item.get("assessments", [])
            if assessments:
                for assessment in assessments:
                    lines.append(
                        f"- `{_text(assessment.get('direction'))}` `{_text(assessment.get('evidence_id'))}` "
                        f"at `{_text(assessment.get('source_path'))}`; LR `{_text(assessment.get('lr'))}`, confidence `{_text(assessment.get('confidence'))}`"
                    )
            else:
                lines.append("- 无可验证评估。")
            lines.append("")
    else:
        lines.extend(["- 未形成可验证事实假设。", ""])

    lines.extend(["## 未验证项", "", *_bullets(report.get("unverified_items", [])), "", "## 信息缺口", "", *_bullets(report.get("information_gaps", [])), "", "## 未选择更高风险标签的原因", ""])
    higher = report.get("why_not_higher_risk", [])
    if higher:
        for item in higher:
            reason = item.get("reason", item.get("unmet_conditions", ""))
            lines.append(f"- 标签 `{_text(item.get('label'))}`: {_text(reason)}")
    else:
        lines.append("- 无。")
    lines.append("")
    return "\n".join(lines)
