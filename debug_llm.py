"""使用项目环境配置执行一次最小、真实的 JSON mode 诊断。"""

from pydantic import BaseModel, ConfigDict

from llm_output import ChatOpenAIAdapter, create_default_llm


class AnswerModel(BaseModel):
    """诊断调用的最小严格响应。"""

    model_config = ConfigDict(extra="forbid")
    answer: str


def main() -> int:
    """仅在显式执行脚本时调用网络，且不输出客户端配置或密钥。"""
    client = create_default_llm()
    if client is None:
        print("未能从项目环境创建 LLM 客户端，请检查配置和可选依赖。")
        return 2

    adapter = ChatOpenAIAdapter(client, structured_method="json_mode")
    result = adapter.invoke_structured(
        "Return one JSON object with exactly one string field named answer.",
        "Explain quantum entanglement in one short paragraph.",
        AnswerModel,
    )
    if not result.ok:
        exception_type = result.failure.exception_type or "None"
        print(
            f"结构化诊断失败：{result.failure.code.value}; "
            f"exception_type={exception_type}"
        )
        return 1

    print("结构化诊断成功。")
    print(f"answer: {result.value.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())