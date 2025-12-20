from typing import Dict, List, Any
from langchain_core.messages import HumanMessage
from agent.core import get_model, setup_agent
from agent.tools_registry import get_tool

model = get_model()
qa_agent = setup_agent()


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

    fetch_messages = await get_tool("fetch_messages")
    build_index = await get_tool("build_index")
    evidence_retrieve = await get_tool("evidence_retrieve")

    # 1) Fetch messages
    md = _md_header(scope) + "⏳ Fetching messages from Telegram...\n"
    yield md

    fetch_result: Dict[str, Any] = await fetch_messages.ainvoke(scope)

    if not fetch_result or fetch_result.get("success") is not True:
        err = (fetch_result or {}).get("error", "Unknown error")
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
    index_result: Dict[str, Any] = await build_index.ainvoke({
        "raw_path": raw_path, "scope": scope
    })

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
    retrieve_result: Dict[str, Any] = await evidence_retrieve.ainvoke({
        "scope": scope,
        "k": 50,
    })

    if not retrieve_result or not retrieve_result.get("success") is not True:
        err = (retrieve_result or {}).get("error", "Unknown error")
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

    init = {"messages": [HumanMessage(content=summarize_prompt)]}

    async for ev in model.astream_events(init, version="v1"):
        if ev.get("event") == "on_chat_model_stream":
            ch = ev.get("data", {}).get("chunk")
            if ch and getattr(ch, "content", None):
                ai_buf += ch.content
                yield base + ai_buf

    yield base + ai_buf


async def qa_stream(query: str, chat: List[Dict], thread_id: str, scope_state: dict):
    chat = (chat or []) + [{"role": "user", "content": query}, {"role": "assistant", "content": ""}]
    ai_idx = len(chat) - 1
    ai_buf = ""

    yield chat

    init = {"messages": [HumanMessage(content=query)]}

    async for ev in qa_agent.astream_events(
        init,
        version="v1",
        config={"configurable": {"thread_id": str(thread_id), "scope_state": scope_state}},
    ):
        if ev.get("event") == "on_chat_model_stream":
            ch = ev.get("data", {}).get("chunk")
            if ch and getattr(ch, "content", None):
                ai_buf += ch.content
                chat[ai_idx]["content"] = ai_buf
                yield chat

    yield chat
