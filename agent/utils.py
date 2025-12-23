import json
from typing import Dict, List, Any
from langchain_core.messages import AIMessage


def parse_mcp_result(result: Any) -> Dict[str, Any]:
    """Parse MCP tool result. Handles dict, list, and wrapped formats."""
    if isinstance(result, dict):
        # Check if wrapped format
        if "type" in result and "text" in result:
            try:
                text_content = result.get("text", "")
                if isinstance(text_content, str):
                    parsed = json.loads(text_content)
                    return parsed
                else:
                    return text_content
            except (json.JSONDecodeError, TypeError):
                return result
        else:
            return result
    elif isinstance(result, list):
        if result:
            return parse_mcp_result(result[0])
        else:
            return {"success": False, "error": "Empty result list"}
    else:
        return {"result": result} if result else {"success": False, "error": "Invalid result format"}


def extract_ai_message_from_chunk(
    chat: List[Dict[str, str]],
    chunk: Any,
    ai_idx: int,
):
    """
    Extract the latest AIMessage content from a LangGraph `astream` chunk
    and update the assistant message in the chat history.

    This function is used as a fallback when `astream_events` is unavailable
    or unreliable. The structure of `astream` output may vary depending on
    the graph, node configuration, or LangGraph version, so multiple shapes
    are handled defensively.
    """

    # astream typically yields a dict: {node_name: node_output}
    if not isinstance(chunk, dict):
        return None

    for node_name, node_output in chunk.items():

        # Case 1: node_output is a list of messages
        if isinstance(node_output, list):
            for msg in node_output:
                if isinstance(msg, AIMessage) and msg.content:
                    chat[ai_idx]["content"] = (
                        msg.content if isinstance(msg.content, str) else str(msg.content)
                    )
                    return chat

        # Case 2: node_output is a single AIMessage
        elif isinstance(node_output, AIMessage) and node_output.content:
            chat[ai_idx]["content"] = (
                node_output.content
                if isinstance(node_output.content, str)
                else str(node_output.content)
            )
            return chat

        # Case 3: node_output is a dict containing a "messages" field
        elif isinstance(node_output, dict):
            messages = node_output.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    chat[ai_idx]["content"] = (
                        msg.content if isinstance(msg.content, str) else str(msg.content)
                    )
                    return chat

    return None


def fmt_scope(scope: Dict[str, Any]) -> str:
    ch = scope.get("channel", "")
    f = scope.get("from", "")
    t = scope.get("to", "")
    return f"- Channel: `{ch}`\n- Time range: `{f}` ~ `{t}`\n"


def md_header(scope: Dict[str, Any]) -> str:
    return "## Summary\n\n" + fmt_scope(scope) + "\n"
