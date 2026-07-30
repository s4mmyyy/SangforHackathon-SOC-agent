"""离线查看 NDR 样例的保守解析结果。"""

import json
from pathlib import Path

from GraphParser import NDRGraphParser


def main(path: str = "NDR_example.json") -> None:
    """读取 JSON 后复用容错解析器，避免调试入口绕过输入校验。"""
    sample_path = Path(path)
    with sample_path.open(encoding="utf-8") as file:
        data = json.load(file)

    parser = NDRGraphParser(data)
    alert = parser.to_structured_alert()
    print(f"原子事实数: {len(alert.atomic_facts)}")
    print(f"结构诊断数: {len(parser.diagnostics)}")
    for fact in alert.atomic_facts:
        print(f"  {fact}")
    for diagnostic in parser.diagnostics:
        print(f"  [诊断] {diagnostic}")


if __name__ == "__main__":
    main()
