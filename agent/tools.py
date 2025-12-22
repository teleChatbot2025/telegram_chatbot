from typing import Dict, Any, Annotated
import json
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

from agent.tools_registry import get_tool


def _parse_mcp_result(result: Any) -> Dict[str, Any]:
    """Parse MCP tool result."""
    if isinstance(result, dict):
        if "type" in result and "text" in result:
            try:
                text_content = result.get("text", "")
                if isinstance(text_content, str):
                    return json.loads(text_content)
                else:
                    return text_content
            except (json.JSONDecodeError, TypeError):
                return result
        else:
            return result
    elif isinstance(result, list):
        if result:
            return _parse_mcp_result(result[0])
        else:
            return {"success": False, "error": "Empty result list"}
    else:
        return {"result": result} if result else {"success": False, "error": "Invalid result format"}


@tool
async def scoped_retrieve(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg()] = None
) -> Dict[str, Any]:
    """
    Retrieve relevant content strictly within the server-defined analysis scope.

    This tool is exposed to the language model with a minimal interface:
    the model can only provide a natural-language query. All contextual
    constraints (such as channel, time range, or index selection) are
    injected at runtime by the system and are not visible or modifiable
    by the model.

    The retrieval scope is obtained from the execution configuration
    (RunnableConfig.configurable["scope_state"]), which is populated
    during the prior analysis phase. If no scope is available, the tool
    will fail and instruct the user to run the analysis step first.

    Parameters
    ----------
    query : str
        The user query describing the information to retrieve.

    Returns
    -------
    Dict[str, Any]
        A JSON-serializable retrieval payload constrained to the pre-defined
        analysis scope. The returned dictionary is expected to follow the MCP
        tool contract, for example:

        - success: bool
          Indicates whether retrieval succeeded.
        - results: list
          A list of retrieved chunks/documents. Each item typically includes
          fields such as `text`, `metadata`, and optionally a relevance score.
        - error: str (optional)
          Present when `success` is False, describing the failure reason.

    Raises
    ------
    ValueError
        If the execution configuration is missing or if no analysis scope
        has been defined for the current session.
    """
    if config is None:
        raise ValueError("Missing RunnableConfig.")

    scope_state = (config.get("configurable") or {}).get("scope_state")
    if not scope_state:
        raise ValueError("No scope found. Please run Analyze first.")

    mcp_retrieve = await get_tool("retrieve")

    result = await mcp_retrieve.ainvoke({
        "query": query,
        "channel": scope_state.get("channel", ""),
        "from_date": scope_state.get("from", ""),
        "to_date": scope_state.get("to", "")
    })

    # Parse MCP result
    return _parse_mcp_result(result)
