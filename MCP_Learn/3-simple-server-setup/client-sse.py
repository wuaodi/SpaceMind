import asyncio
import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
import os

# 忽略clash代理设置，确保可以访问localhost，我已经设置在了系统的环境变量里面了
# os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# nest_asyncio.apply()  # Needed to run interactive python

"""
Make sure:
1. The server is running before running this script.
2. The server is configured to use SSE transport.
3. The server is listening on port 8050.

To run the server:
uv run server.py
"""


async def main():
    # Connect to the server using SSE
    async with sse_client("http://localhost:8050/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            # initialize点开看会发现是异步定义的，所以需要加await，不加的话只是创建协程而不会执行
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Call our calculator tool
            result = await session.call_tool("add", arguments={"a": 2, "b": 3})
            print(f"2 + 3 = {result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
