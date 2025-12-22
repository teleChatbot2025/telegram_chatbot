from typing import Dict, List, Any
import json
from langchain_core.messages import HumanMessage, AIMessage
from agent.core import get_model, setup_agent
from agent.tools_registry import get_tool


def _parse_mcp_result(result: Any) -> Dict[str, Any]:
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
            return _parse_mcp_result(result[0])
        else:
            return {"success": False, "error": "Empty result list"}
    else:
        return {"result": result} if result else {"success": False, "error": "Invalid result format"}


def _fmt_scope(scope: Dict[str, Any]) -> str:
    ch = scope.get("channel", "")
    f = scope.get("from", "")
    t = scope.get("to", "")
    return f"- Channel: `{ch}`\n- Time range: `{f}` ~ `{t}`\n"


def _md_header(scope: Dict[str, Any]) -> str:
    return "## Summary\n\n" + _fmt_scope(scope) + "\n"


async def analyze_stream(scope: dict):
    """
    Stream a Markdown summary for the given analysis scope.

    This function simulates an analysis workflow and streams the
    generated summary in Markdown format.
    """
    md = _md_header(scope) + "⏳ Initializing analysis...\n"
    yield md

    try:
        fetch_messages = await get_tool("fetch_messages")
        build_index = await get_tool("build_index")
        evidence_retrieve = await get_tool("evidence_retrieve")
    except ConnectionError as e:
        md = _md_header(scope) + f"❌ **MCP server connection failed**\n\n{str(e)}\n"
        yield md
        return
    except KeyError as e:
        md = _md_header(scope) + (
            f"❌ **Tool not found**: {str(e)}\n\n"
            "Ensure MCP server provides required tools:\n"
            "- fetch_messages\n"
            "- build_index\n"
            "- evidence_retrieve\n"
        )
        yield md
        return
    except Exception as e:
        md = _md_header(scope) + f"❌ **Initialization failed**: {str(e)}\n"
        yield md
        return

    # 1) Fetch messages
    md = _md_header(scope) + "⏳ Fetching messages from Telegram...\n"
    yield md

    # FastMCP requires unpacking dict params
    fetch_result = await fetch_messages.ainvoke({
        "channel": scope.get("channel", ""),
        "from_date": scope.get("from", ""),
        "to_date": scope.get("to", "")
    })

    # Parse MCP result (may be wrapped format)
    fetch_result = _parse_mcp_result(fetch_result)

    if not fetch_result:
        md = _md_header(scope) + "❌ Fetch failed: Empty result\n"
        yield md
        return
    
    # Check success field
    if fetch_result.get("success") is not True:
        err = fetch_result.get("error", "Unknown error")
        md = _md_header(scope) + f"❌ Fetch failed: `{err}`\n"
        yield md
        return

    raw_path = fetch_result.get("path")
    msg_count = fetch_result.get("count", 0)

    md = _md_header(scope) + (
        f"✅ Fetched **{msg_count}** messages.\n\n"
        f"- Saved to: `{raw_path}`\n\n"
        "⏳ Building vector index...\n"
    )
    yield md

    # 2) Build index
    index_result = await build_index.ainvoke({
        "raw_path": raw_path,
        "channel": scope.get("channel", ""),
        "from_date": scope.get("from", ""),
        "to_date": scope.get("to", "")
    })

    # Parse MCP result
    index_result = _parse_mcp_result(index_result)

    if not index_result or index_result.get("success") is not True:
        err = (index_result or {}).get("error", "Unknown error")
        md = _md_header(scope) + f"❌ Index build failed: `{err}`\n"
        yield md
        return

    chunks = index_result.get("chunks", 0)

    md = _md_header(scope) + (
        f"✅ Vector index ready.\n\n"
        f"- Chunks added: **{chunks}**\n"
        "⏳ Retrieving evidence chunks...\n"
    )
    yield md

    # 3) Retrieve evidence for summary
    retrieve_result = await evidence_retrieve.ainvoke({
        "k": 50,
        "channel": scope.get("channel", ""),
        "from_date": scope.get("from", ""),
        "to_date": scope.get("to", "")
    })

    # Parse MCP result
    retrieve_result = _parse_mcp_result(retrieve_result)

    if not retrieve_result:
        md = _md_header(scope) + "❌ Retrieve failed: Empty result\n"
        yield md
        return
    
    if retrieve_result.get("success") is not True:
        err = retrieve_result.get("error", "Unknown error")
        if err == "Unknown error":
            if "result" in retrieve_result:
                err = f"Result: {str(retrieve_result.get('result'))[:200]}"
            elif isinstance(retrieve_result, dict) and len(retrieve_result) > 0:
                err = f"Data: {str(retrieve_result)[:200]}"
        md = _md_header(scope) + f"❌ Retrieve failed: `{err}`\n"
        yield md
        return

    evidences = retrieve_result.get("evidences", []) or []
    k = len(evidences)

    evidence_lines = []
    for i, r in enumerate(evidences, start=1):
        text = ((r.get("text") or "").strip())[:512]
        meta = r.get("metadata") or {}
        ts = meta.get("timestamp")
        mid = meta.get("message_id")
        evidence_lines.append(f"[{i}] {text}\n(meta: timestamp={ts}, message_id={mid})")

    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "(No evidence retrieved)"

    md = _md_header(scope) + (
        f"✅ Retrieved **{k}** evidence chunks.\n\n"
        "⏳ Generating summary (streaming)...\n\n"
    )
    yield md

    # 4) LLM streaming summary
    summarize_prompt = (
        "You are a Telegram chat analyzer.\n"
        "Write a structured Markdown summary for the given scope.\n\n"
        "Requirements:\n"
        "- Use headings and bullet points\n"
        "- Cover main topics, key events, and conclusions\n"
        "- Mention limitations (this is based on retrieved evidence)\n\n"
        f"Scope:\n{_fmt_scope(scope)}\n"
        "Evidence:\n"
        f"{evidence_block}\n"
    )

    ai_buf = ""
    base = _md_header(scope)

    messages = [HumanMessage(content=summarize_prompt)]

    model = get_model()
    # Stream LLM response
    async for chunk in model.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            ai_buf += chunk.content
            yield base + ai_buf

    yield base + ai_buf


async def qa_stream(query: str, chat: List[Dict], thread_id: str, scope_state: dict):
    chat = (chat or []) + [{"role": "user", "content": query}, {"role": "assistant", "content": ""}]
    ai_idx = len(chat) - 1
    ai_buf = ""

    yield chat

    input_messages = [HumanMessage(content=query)]
    config = {"configurable": {"thread_id": str(thread_id), "scope_state": scope_state}}

    qa_agent = setup_agent()
    
    # Use astream_events for streaming (more reliable)
    try:
        async for event in qa_agent.astream_events(
            {"messages": input_messages},
            version="v1",
            config=config
        ):
            if event.get("event") == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    ai_buf += chunk.content
                    chat[ai_idx]["content"] = ai_buf
                    yield chat
            elif event.get("event") == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if output and hasattr(output, "content") and output.content:
                    ai_buf = output.content
                    chat[ai_idx]["content"] = ai_buf
                    yield chat
    except Exception as e:
        # Fallback to astream
        print(f"⚠️ astream_events failed, trying astream: {e}")
        try:
            async for chunk in qa_agent.astream(
                {"messages": input_messages},
                config=config
            ):
                # Process agent output: {node_name: [messages]}
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        if isinstance(node_output, list):
                            for msg in node_output:
                                if isinstance(msg, AIMessage):
                                    if hasattr(msg, "content") and msg.content:
                                        ai_buf = msg.content if isinstance(msg.content, str) else str(msg.content)
                                        chat[ai_idx]["content"] = ai_buf
                                        yield chat
                        elif isinstance(node_output, AIMessage):
                            if hasattr(node_output, "content") and node_output.content:
                                ai_buf = node_output.content if isinstance(node_output.content, str) else str(node_output.content)
                                chat[ai_idx]["content"] = ai_buf
                                yield chat
                        elif isinstance(node_output, dict):
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                if isinstance(msg, AIMessage):
                                    if hasattr(msg, "content") and msg.content:
                                        ai_buf = msg.content if isinstance(msg.content, str) else str(msg.content)
                                        chat[ai_idx]["content"] = ai_buf
                                        yield chat
        except Exception as e2:
            error_msg = f"Agent execution failed: {str(e2)}"
            chat[ai_idx]["content"] = f"❌ {error_msg}\n\nPlease check:\n1. DEEPSEEK_API_TOKEN is set\n2. Network connection\n3. Check console logs"
            yield chat
            print(f"❌ Agent error: {e2}")
            import traceback
            traceback.print_exc()

    yield chat
