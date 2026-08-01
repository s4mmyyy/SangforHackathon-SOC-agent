import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# 1. 定义你的 Pydantic 模型（示例：包含一个 answer 字段）
class AnswerModel(BaseModel):
    answer: str  # 要求返回一个字符串类型的答案

# 2. 初始化客户端（保持原有配置）
client = OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key="sk-84c00081dec5433e8ac0d2af0669d508",
)

# 3. 构建请求，开启 JSON 模式并明确提示返回格式
try:
    completion = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": "You must respond with a valid JSON object containing a field 'answer' of type string."
            },
            {
                "role": "user",
                "content": "Explain quantum entanglement in one paragraph."
            }
        ],
        response_format={"type": "json_object"},  # 强制返回 JSON
        temperature=0.3,  # 降低随机性，提高输出稳定性
    )

    # 4. 获取返回的 JSON 字符串
    raw_content = completion.choices[0].message.content
    print("原始返回内容：", raw_content)

    # 5. 解析 JSON 并用 Pydantic 校验
    data = json.loads(raw_content)
    validated = AnswerModel(**data)  # 如果字段不匹配会抛出 ValidationError

    print("\n✅ 校验成功！")
    print(f"answer 字段内容：{validated.answer}")

except json.JSONDecodeError as e:
    print(f"❌ JSON 解析失败：{e}")
except ValidationError as e:
    print(f"❌ Pydantic 校验失败：{e.json()}")
except Exception as e:
    print(f"❌ API 调用或其他错误：{e}")