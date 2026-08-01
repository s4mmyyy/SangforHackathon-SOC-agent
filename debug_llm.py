from openai import OpenAI

client = OpenAI(
    base_url="https://lingsuan.top/v1",
    api_key="sk-c2b8fbc58acce0c27991b3c5a3127675e98de8a7bacf7f45109fd9c780c04fe7",
)

completion = client.chat.completions.create(
    model="gpt-5.4",
    messages=[
        {"role": "user", "content": "Explain quantum entanglement in one paragraph."}
    ],
)

print(completion.choices[0].message.content)