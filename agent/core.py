import os
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from agent.prompt import SYSTEM_PROMPT
from agent.tools import scoped_retrieve

_model = None
_agent = None


def get_model():
    global _model
    if _model is None:
        api_key = os.getenv("DEEPSEEK_API_TOKEN")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_TOKEN in environment.")
        _model = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            temperature=0,
        )
    return _model


def setup_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(
            model=get_model(),
            tools=[scoped_retrieve],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
            name="telegram_chatbot",
        )
    return _agent
