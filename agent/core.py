import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from agent.prompt import QA_PROMPT, SUMMARY_PROMPT
from agent.tools import scoped_retrieve
from agent.utils import fmt_scope

_model = None
_agent = None
_summary_chain = None


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


def get_summary_chain():
    """
    Returns a runnable chain:
      inputs: {"scope": dict, "evidence_block": str}
      output: streamed AIMessageChunk from model
    """
    global _summary_chain
    if _summary_chain is None:
        # 1) map -> render a single HumanMessage(prompt)
        to_messages = RunnableLambda(
            lambda x: [
                HumanMessage(
                    content=SUMMARY_PROMPT.format(
                        scope_md=fmt_scope(x["scope"]),
                        evidence=x["evidence_block"],
                    )
                )
            ]
        )

        # 2) messages -> model
        _summary_chain = to_messages | get_model()
    return _summary_chain


def setup_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(
            model=get_model(),
            tools=[scoped_retrieve],
            system_prompt=QA_PROMPT,
            checkpointer=MemorySaver(),
            name="telegram_chatbot",
        )
    return _agent
