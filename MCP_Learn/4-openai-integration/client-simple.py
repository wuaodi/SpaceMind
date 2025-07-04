import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# Apply nest_asyncio to allow nested event loops (needed for Jupyter/IPython)
nest_asyncio.apply()

# Load environment variables
load_dotenv("../.env")

# Global variables to store session state
session = None
exit_stack = AsyncExitStack()
model = "gpt-4o"
stdio = None
write = None
openai_client = AsyncOpenAI(
            api_key="sk-live-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJNZXRhQ2hhdCIsInN1YiI6IjY4NjY5MTVkNWQxYjY1YWExNzVmMmY1OSIsImNsaWVudF9pZCI6ImU3NjA1YTA2NzRmMGY5NWZhMjI3MjRkMjIyMWNlMTFjIiwic2NvcGUiOiJtaWRqb3VybmV5IGFnZW50IiwiaWF0IjoxNzUxNTUzMTQ0fQ.6g6RB_ieV3x5VwoSLqtx-uDyMIBhut2VhOjIo7Olf9I",
            base_url="https://llm-api.mmchat.xyz/v1"
        )

async def connect_to_server(server_script_path: str = "server.py"):
    """Connect to an MCP server.

    Args:
        server_script_path: Path to the server script.
    """
    global session, stdio, write, exit_stack

    # Server configuration
    server_params = StdioServerParameters(
        command="python",
        args=[server_script_path],
    )

    # Connect to the server
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    # Initialize the connection
    await session.initialize()

    # List available tools
    tools_result = await session.list_tools()
    print("\nConnected to server with tools:")
    for tool in tools_result.tools:
        print(f"  - {tool.name}: {tool.description}")


async def get_mcp_tools() -> List[Dict[str, Any]]:
    """Get available tools from the MCP server in OpenAI format.

    Returns:
        A list of tools in OpenAI format.
    """
    global session

    tools_result = await session.list_tools()
    
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools_result.tools
    ]


async def process_query(query: str) -> str:
    """Process a query using OpenAI and available MCP tools.

    Args:
        query: The user query.

    Returns:
        The response from OpenAI.
    """
    global session, openai_client, model

    # Get available tools
    tools = await get_mcp_tools()

    # Initial OpenAI API call

    
    # 添加系统提示，会大大增加调用工具的准确性
    system_message = {
        "role": "system", 
        "content": "你是一个有用的AI助手，可以使用工具来帮助用户。当用户询问问题时，请根据需要使用可用的工具来获取信息，然后基于工具返回的结果来回答用户的问题。"
    }
    print(f"message_query: {system_message, {'role': 'user', 'content': query}}")
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[system_message, {"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
    )
    
    print(f"assistant_message: {response.choices[0].message}")
    # Get assistant's response
    assistant_message = response.choices[0].message

    # Initialize conversation with user query and assistant response
    messages = [
        {"role": "user", "content": query},
        assistant_message,
    ]
    print(f"messages_add_assistant_message: {messages}")

    # Handle tool calls if present
    if assistant_message.tool_calls:
        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            # Execute tool call
            print(f"Tool call: {tool_call.function.name}")
            print(f"Tool call arguments: {tool_call.function.arguments}")
            result = await session.call_tool(
                tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            )

            # Add tool response to conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content[0].text,
                }
            )


        print(f"messages_before_final_response: {messages}")
        
        # Get final response from OpenAI with tool results
        final_response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="none",  # Don't allow more tool calls
        )
        print(f"final_message: {final_response.choices[0].message}")

        return final_response.choices[0].message.content

    # No tool calls, just return the direct response
    return assistant_message.content


async def cleanup():
    """Clean up resources."""
    global exit_stack
    await exit_stack.aclose()


async def main():
    """Main entry point for the client."""
    await connect_to_server("server.py")

    # Example: Ask about company vacation policy
    query = "What is our company's vacation policy?"
    print(f"\nQuery: {query}")

    response = await process_query(query)
    print(f"\nResponse: {response}")

    await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
