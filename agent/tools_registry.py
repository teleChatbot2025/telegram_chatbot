import asyncio
from typing import Dict, List, Optional, Any
from langchain_mcp_adapters.client import MultiServerMCPClient

_tools: Optional[List[Any]] = None
_tool_map: Optional[Dict[str, Any]] = None
_lock = asyncio.Lock()


async def load_tools() -> List[Any]:
    client = MultiServerMCPClient({
        "local": {
            "url": "http://127.0.0.1:22331/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()
    return tools


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
