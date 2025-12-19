import asyncio
from typing import Dict, List
from langchain_core.messages import HumanMessage
from agent.core import setup_agent

qa_agent = asyncio.run(setup_agent())


async def analyze_stream(scope: dict):
    raise NotImplementedError


async def qa_stream(query: str, chat: List[Dict], thread_id: str):
    chat = (chat or []) + [{"role": "user", "content": query}, {"role": "assistant", "content": ""}]
    ai_idx = len(chat) - 1
    ai_buf = ""

    yield chat

    init = {"messages": [HumanMessage(content=query)]}

    async for ev in qa_agent.astream_events(
        init,
        version="v1",
        config={"configurable": {"thread_id": str(thread_id)}},
    ):
        if ev.get("event") == "on_chat_model_stream":
            ch = ev.get("data", {}).get("chunk")
            if ch and getattr(ch, "content", None):
                ai_buf += ch.content
                chat[ai_idx]["content"] = ai_buf
                yield chat

    yield chat
