import uuid
import gradio as gr
from agent.runner import qa_stream, analyze_stream

DEFAULT_SUMMARY_MD = "## Summary\n\n(Click **Analyze** to generate summary)"


def clear_chat_history():
    """Clear only chat history; keep summary/scope unchanged."""
    chatbox = []
    msg = ""
    chat_state = []
    session_id = str(uuid.uuid4())
    return chatbox, msg, chat_state, session_id


def reset_all():
    """Reset BOTH sides: summary + scope + chat + session + input box."""
    summary_md = DEFAULT_SUMMARY_MD
    chatbox = []
    msg = ""
    chat_state = []
    scope_state = {}
    session_id = str(uuid.uuid4())
    return summary_md, chatbox, msg, chat_state, scope_state, session_id


async def on_submit(user_msg, chat_state, scope_state, session_id):
    """
    Handle the Q&A (right panel) as an async generator for streaming updates.

    Inputs
    - user_msg (str): User question from the right input box.
    - chat_state (list[dict]): Current chat history in Gradio "messages" format.
    - scope_state (dict): Analysis scope produced by the left Analyze step.
    - session_id (str): LangChain thread_id for conversation memory.

    Outputs (streamed via yield)
    - chatbox_value (list[dict]): Updated messages for gr.Chatbot(type="messages").
    - chat_state (list[dict]): Same messages persisted in gr.State for next turn.

    Behavior
    - If scope_state is empty: show a warning and keep UI/state unchanged.
    - Otherwise: stream tokens from qa_stream and continuously update chatbox/chat_state.
    """
    if not scope_state:
        gr.Warning("Please run analyze before proceeding with the Q&A.")
        yield chat_state, chat_state
        return

    async for chat in qa_stream(user_msg, chat_state, session_id):
        yield chat, chat


async def on_analyze(channel, date_from, date_to, summary_md_current, scope_state):
    """
    Handle the analysis (left panel) as an async generator for streaming summary updates.

    Inputs
    - channel (str): Telegram channel identifier (e.g. @some_channel).
    - date_from (str): Start date string (YYYY-MM-DD).
    - date_to (str): End date string (YYYY-MM-DD).
    - summary_md_current (str): Current markdown shown in the summary panel.
    - scope_state (dict): Previously analyzed scope.

    Outputs (streamed via yield)
    - summary_md (str): Markdown string to render in the summary panel.
    - scope_state (dict): Updated scope dict: {"channel": ..., "from": ..., "to": ...}

    Behavior
    - Validate inputs; if missing, warn and keep summary/scope unchanged.
    - If the new scope equals the existing scope, warn and keep summary/scope unchanged.
    - Otherwise, stream markdown from analyze_stream(new_scope) and update scope_state.
    """
    new_scope = {"channel": channel, "from": date_from, "to": date_to}

    if not channel or not date_from or not date_to:
        gr.Warning("Please provide channel, From, and To before analyzing.")
        yield summary_md_current, scope_state
        return

    if scope_state == new_scope:
        gr.Warning("The analysis scope remains unchanged, so there is no need to repeat the analyze.")
        yield summary_md_current, scope_state
        return

    async for summary_md in analyze_stream(new_scope):
        yield summary_md, new_scope


def build_ui():
    with gr.Blocks(title="Telegram Analyzer WebUI", css=open("ui/styles.css").read()) as demo:
        # ---- States ----
        chat_state = gr.State([])
        scope_state = gr.State({})
        session_id = gr.State()
        demo.load(fn=lambda: str(uuid.uuid4()), outputs=session_id)

        gr.Markdown("# ⚙️ Telegram Analyzer WebUI")

        with gr.Row(equal_height=True, elem_id="layout_main_row"):
            # Left: Summary + Analyze inputs
            with gr.Column(scale=1, min_width=360, elem_id="left_column"):
                summary_md = gr.Markdown(value=DEFAULT_SUMMARY_MD, elem_id="summary_view")

                with gr.Column(elem_id="left_panel"):
                    with gr.Row():
                        channel = gr.Textbox(label="Channel", placeholder="e.g. @some_channel", lines=1, max_lines=1)
                    with gr.Row():
                        date_from = gr.Textbox(label="From", placeholder="YYYY-MM-DD", lines=1, max_lines=1)
                        date_to = gr.Textbox(label="To", placeholder="YYYY-MM-DD", lines=1, max_lines=1)
                    with gr.Row():
                        analyze_btn = gr.Button("🧠 Analyze", variant="primary")
                        reset_btn = gr.Button("🔄 Reset")

            # Right: Chat QA
            with gr.Column(scale=2, min_width=720, elem_id="right_column"):
                chatbox = gr.Chatbot(elem_id="chat_view", type="messages", show_label=False, height="100%")
                with gr.Column(elem_id="right_panel"):
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Enter your question...", show_label=False, lines=2, max_lines=2)
                    with gr.Row():
                        send_btn = gr.Button("🚀 Send", variant="primary")
                        clear_chat_btn = gr.Button("🧹 Clear chat history")

        # ---- Left events ----
        analyze_btn.click(
            on_analyze,
            inputs=[channel, date_from, date_to, summary_md, scope_state],
            outputs=[summary_md, scope_state],
        )

        reset_btn.click(
            reset_all,
            inputs=None,
            outputs=[summary_md, chatbox, msg, chat_state, scope_state, session_id],
        )

        # ---- Right events ----
        clear_chat_btn.click(
            clear_chat_history,
            inputs=None,
            outputs=[chatbox, msg, chat_state, session_id],
        )

        staged = gr.State("")
        msg.submit(
            lambda x: x, inputs=msg, outputs=staged,
        ).then(
            lambda: "", inputs=None, outputs=msg,
        ).then(
            on_submit,
            inputs=[staged, chat_state, scope_state, session_id],
            outputs=[chatbox, chat_state],
        )

        send_btn.click(
            lambda x: x, inputs=msg, outputs=staged,
        ).then(
            lambda: "", inputs=None, outputs=msg,
        ).then(
            on_submit,
            inputs=[staged, chat_state, scope_state, session_id],
            outputs=[chatbox, chat_state],
        )

    return demo
