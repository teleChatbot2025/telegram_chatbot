import asyncio
from typing import Dict, List, Optional, Any
from langchain_mcp_adapters.client import MultiServerMCPClient

_tools: Optional[List[Any]] = None
_tool_map: Optional[Dict[str, Any]] = None
_lock = asyncio.Lock()

MCP_SERVER_URL = "http://127.0.0.1:22331/mcp"


async def load_tools() -> List[Any]:
    """Load MCP tools. Raises ConnectionError if connection fails."""
    try:
        client = MultiServerMCPClient({
            "local": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        })
        tools = await client.get_tools()
        return tools
    except Exception as e:
        error_msg = (
            f"Failed to connect to MCP server: {MCP_SERVER_URL}\n\n"
            f"Error: {str(e)}\n\n"
            "Please ensure:\n"
            "1. MCP server is running\n"
            "2. Server address and port are correct\n"
            "3. Firewall is not blocking the connection"
        )
        raise ConnectionError(error_msg) from e


async def get_tools() -> List[Any]:
    global _tools, _tool_map
    if _tools is not None:
        return _tools
    async with _lock:
        if _tools is None:
            _tools = await load_tools()
            _tool_map = {t.name: t for t in _tools}
    return _tools


async def get_tool(name: str) -> Any:
    await get_tools()
    assert _tool_map is not None
    return _tool_map[name]
