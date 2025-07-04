import json

import openai
from dotenv import load_dotenv
from tools import add

load_dotenv("../.env")

"""
This is a simple example to demonstrate that MCP simply enables a new way to call functions.
"""

# Define tools for the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


# 创建 OpenAI 客户端，使用第三方集成的 API
api_key = "sk-live-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJNZXRhQ2hhdCIsInN1YiI6IjY4NjY5MTVkNWQxYjY1YWExNzVmMmY1OSIsImNsaWVudF9pZCI6ImU3NjA1YTA2NzRmMGY5NWZhMjI3MjRkMjIyMWNlMTFjIiwic2NvcGUiOiJtaWRqb3VybmV5IGFnZW50IiwiaWF0IjoxNzUxNTUzMTQ0fQ.6g6RB_ieV3x5VwoSLqtx-uDyMIBhut2VhOjIo7Olf9I"
base_url = "https://llm-api.mmchat.xyz/v1"
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url
)

# 增加系统提示
system_message = {
    "role": "system",
    "content": "你是一个有用的AI助手，可以使用工具来帮助用户。\
    当用户询问问题时，请根据需要使用可用的工具来获取信息，然后基于工具返回的结果来回答用户的问题。"
}

# 增加用户提示
user_message = {
    "role": "user",
    "content": "帮我计算 25 + 17"
}

# Call LLM
response = client.chat.completions.create(
    model="gpt-4o",  # 使用 Metachat 支持的模型
    messages=[system_message, 
              user_message],
    tools=tools,
    tool_choice="auto",
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    print(f"调用工具的名称: {response.choices[0].message.tool_calls[0].function.name}")
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    # Execute directly
    result = add(**tool_args)

    # Send result back to model
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "帮我计算 25 + 17"},
            response.choices[0].message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)},
        ],
        tools=tools,
        tool_choice="auto"
    )
    print(final_response.choices[0].message.content)
else:
    print(f"没有调用工具，直接返回结果")
    print(response.choices[0].message.content)
