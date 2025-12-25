# Telegram Chat Analyzer (Gradio + LangChain + MCP)

A small course project that analyzes Telegram messages within a selected scope (channel + time range) and provides:
1) **Streaming Summary** of the selected period  
2) **Q&A Assistant** that answers questions based on the analyzed data

This project demonstrates a clean separation between **analysis workflow** and **interactive Q&A**, and uses **MCP (Model Context Protocol)** to expose backend services to the UI/agent layer.

---

## ✨ Features

### 1) Analyze (Left Panel)
- Select a Telegram channel and a date range
- Fetch messages from Telegram (backend tool)
- Build vector index from the fetched messages (backend tool)
- Retrieve representative evidence chunks (backend tool)
- Stream a structured Markdown summary to the UI (LLM streaming)

### 2) Q&A (Right Panel)
- Ask questions after the analysis step
- A ReAct-style agent answers questions
- The agent has access to a single tool: `scoped_retrieve`
- Retrieval is **scope-aware** but the scope is **not visible to the LLM**
- Answers are returned with streaming

---

## 🧱 Architecture Overview

### Components
- **Web UI**: Gradio interface for Analyze + Chat Q&A  
- **MCP Server**: FastMCP server exposing backend tools (fetch/index/retrieve)  
- **Agent Layer**: LangChain/LangGraph agent for Q&A  
- **Scope State**: Stored in Gradio `State`, injected into agent runtime config

### Data Flow

#### Analyze Flow
1. UI collects `scope = {channel, from, to}`
2. Calls MCP tool `fetch_messages(scope)` → returns local file path + count
3. Calls MCP tool `build_index(raw_path, scope)` → returns chunk count
4. Calls MCP tool `evidence_retrieve(k, scope)` → returns evidence list
5. Runs LLM summarization and streams Markdown summary to UI

#### Q&A Flow
1. User asks a question
2. Agent calls `scoped_retrieve(query)` if needed  
3. `scoped_retrieve` injects the scope from runtime config (not visible to LLM)
4. MCP tool `retrieve(scope, query)` returns results
5. Agent answers in streaming mode

---

## 🚀 Run

One-command startup

```
python webui.py
```

This will:
1. Start MCP server if not running
2. Wait until MCP server is ready
3. Launch Gradio Web UI

Web UI will be available at:
- http://0.0.0.0:22337

MCP server endpoint:
- http://127.0.0.1:22331/mcp

---

## 🧰 MCP Tools Contract

The MCP server exposes backend tools under these names:

### `fetch_messages(scope)`
Fetch Telegram messages within scope and save them locally.

### `build_index(raw_path, scope)`
Read raw messages, chunk them, embed them, and build a vector index.

### `evidence_retrieve(k, scope)`
Retrieve representative evidence chunks for summarization.

### `retrieve(query, scope)`
Retrieve relevant chunks for Q&A.

---

## 🧠 Agent & Tool Wrapping

### `scoped_retrieve(query)`
A LangChain tool exposed to the LLM.

- The LLM can only provide: `query`
- Scope is injected at runtime via `RunnableConfig.configurable["scope_state"]`
- Scope is **not visible to the model** (prevents prompt injection)

If scope is missing, the tool instructs the user to run Analyze first.

---
## 👥 Contributors

- **@lhl4971** — UI development and agentic workflow integration (Gradio + LangChain/LangGraph).
- **@buddyki** — Telegram data collection scripts and MCP tool/server development.
- **@lilyandbill** — Vector database implementation and RAG optimization (chunking, indexing, retrieval quality).

---

## 📌 License

This is a course project for educational purposes.