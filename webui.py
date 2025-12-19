import asyncio
from ui.layout import build_ui


async def main():
    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=22337,
        auth_message="Telegram Chat Analyzer",
        show_api=False
    )

if __name__ == "__main__":
    asyncio.run(main())
