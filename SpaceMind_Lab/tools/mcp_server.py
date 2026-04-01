"""
MCP Tool Server for SpaceMind - communicates with the runtime environment via Redis.
Usage: python -m tools.mcp_server
"""

from mcp.server.fastmcp import FastMCP

from .server_tools import register_aux_tools, register_env_tools, register_sensor_tools

mcp = FastMCP(name="SpaceMind Tools", host="0.0.0.0", port=8050)

register_env_tools(mcp)
register_sensor_tools(mcp)
register_aux_tools(mcp)


if __name__ == "__main__":
    mcp.run(transport="stdio")
