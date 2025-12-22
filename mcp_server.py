"""
MCP Server - Telegram message analysis tools

Features:
1. fetch_messages - Fetch and save messages from Telegram
2. build_index - Build vector index
3. evidence_retrieve - Retrieve evidence (for summary)
4. retrieve - Retrieve (for Q&A)

Usage:
    python mcp_server.py
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from fastmcp import FastMCP
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

# Text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback to simple string splitter
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
        
        def split_text(self, text: str) -> List[str]:
            """Simple text splitter implementation."""
            if not text:
                return []
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
            return chunks

load_dotenv()

mcp = FastMCP("telegram-analyzer-tools")

# Paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEXES_DIR = Path("indexes")
INDEXES_DIR.mkdir(exist_ok=True)

# Vector DB client
chroma_client = chromadb.PersistentClient(
    path=str(INDEXES_DIR / "chroma_db"),
    settings=Settings(anonymized_telemetry=False)
)

# Embedding model (prefer OpenAI, fallback to local)
embedding_model = None
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        print("Using OpenAI Embeddings")
except Exception as e:
    print(f"Failed to initialize OpenAI Embeddings: {e}")

if embedding_model is None:
    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Using HuggingFace Embeddings (all-MiniLM-L6-v2)")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize embedding model: {e}\n"
            "Please install sentence-transformers: pip install sentence-transformers"
        )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)


def get_scope_id(scope: Dict[str, Any]) -> str:
    """Generate unique ID from scope."""
    channel = scope.get("channel", "unknown")
    date_from = scope.get("from", "")
    date_to = scope.get("to", "")
    return f"{channel}_{date_from}_{date_to}".replace("@", "").replace("/", "_")


async def fetch_telegram_messages(scope: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch messages from Telegram using Telethon."""
    from datetime import datetime
    from telethon import TelegramClient
    from telethon.errors import SessionPasswordNeededError
    
    channel = scope.get("channel", "")
    date_from_str = scope.get("from", "")
    date_to_str = scope.get("to", "")
    
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    
    if not api_id or not api_hash:
        raise ValueError(
            "Missing Telegram API credentials. Set environment variables:\n"
            "- TELEGRAM_API_ID\n"
            "- TELEGRAM_API_HASH\n"
            "Get them at: https://my.telegram.org/apps"
        )
    
    try:
        date_from = datetime.strptime(date_from_str, "%Y-%m-%d")
        date_to = datetime.strptime(date_to_str, "%Y-%m-%d")
        date_to = date_to.replace(hour=23, minute=59, second=59)
    except ValueError as e:
        raise ValueError(f"Invalid date format, expected YYYY-MM-DD: {e}")
    
    if date_from > date_to:
        raise ValueError("Start date cannot be later than end date")
    
    session_name = "telegram_session"
    client = TelegramClient(session_name, int(api_id), api_hash)
    
    try:
        await client.connect()
        
        if not await client.is_user_authorized():
            raise ValueError(
                "Telegram client not authorized.\n"
                "Run auth script first:\n"
                "  python scripts/auth_telegram.py\n"
                "Or run manually:\n"
                "  python -c \"from telethon import TelegramClient; "
                "import os, asyncio; from dotenv import load_dotenv; "
                "load_dotenv(); "
                "async def auth(): "
                "client = TelegramClient('telegram_session', "
                "int(os.getenv('TELEGRAM_API_ID')), os.getenv('TELEGRAM_API_HASH')); "
                "await client.start(); print('Auth successful!'); "
                "await client.disconnect(); "
                "asyncio.run(auth())\""
            )
        
        messages = []
        print(f"Fetching messages from {channel} ({date_from_str} to {date_to_str})...")
        
        # Use reverse=True to iterate from old to new
        async for message in client.iter_messages(channel, reverse=True):
            msg_date = message.date.replace(tzinfo=None)
            
            if msg_date < date_from:
                continue
            elif msg_date > date_to:
                break
            
            text = ""
            if message.text:
                text = message.text
            elif message.raw_text:
                text = message.raw_text
            
            sender_id = None
            sender_username = None
            if message.sender:
                sender_id = message.sender.id
                sender_username = getattr(message.sender, 'username', None)
            
            messages.append({
                "id": message.id,
                "text": text,
                "date": message.date.isoformat(),
                "timestamp": int(message.date.timestamp()),
                "sender_id": sender_id,
                "sender_username": sender_username,
                "message_type": "text",
            })
        
        print(f"Successfully fetched {len(messages)} messages")
        return messages
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Telegram messages: {str(e)}")
    finally:
        await client.disconnect()


@mcp.tool(name="fetch_messages")
async def fetch_messages(
    channel: str,
    from_date: str,  # Use from_date to avoid Python keyword conflict
    to_date: str
) -> Dict[str, Any]:
    """
    Fetch raw message data from Telegram and save to local storage.
    
    Args:
    - channel: Channel or group identifier (e.g. @channel_name)
    - from_date: Start date (format: YYYY-MM-DD)
    - to_date: End date (format: YYYY-MM-DD)
    """
    try:
        scope = {
            "channel": channel,
            "from": from_date,
            "to": to_date
        }
        
        if not channel or not from_date or not to_date:
            return {
                "success": False,
                "error": "Missing required parameters: channel, from_date, to_date"
            }
        
        messages = await fetch_telegram_messages(scope)
        
        scope_id = get_scope_id(scope)
        output_path = DATA_DIR / f"{scope_id}_messages.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "path": str(output_path),
            "count": len(messages)
        }
        
    except Exception as e:
        import traceback
        error_detail = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ fetch_messages error: {error_detail}")
        print(f"Traceback:\n{error_trace}")
        return {
            "success": False,
            "error": f"Failed to fetch messages: {error_detail}"
        }


@mcp.tool(name="build_index")
async def build_index(
    raw_path: str,
    channel: str,
    from_date: str,
    to_date: str
) -> Dict[str, Any]:
    """
    Chunk, vectorize, and build searchable vector index from raw messages.
    
    Args:
    - raw_path: Path to raw message file
    - channel: Channel identifier
    - from_date: Start date
    - to_date: End date
    """
    try:
        scope = {
            "channel": channel,
            "from": from_date,
            "to": to_date
        }
        
        if not os.path.exists(raw_path):
            return {
                "success": False,
                "error": f"File not found: {raw_path}"
            }
        
        with open(raw_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        
        if not messages:
            return {
                "success": False,
                "error": "Message data is empty"
            }
        
        texts = []
        metadatas = []
        
        for msg in messages:
            text = msg.get("text", "")
            if not text:
                continue
            
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "message_id": msg.get("id", ""),
                    "timestamp": msg.get("date", ""),
                    "sender_id": msg.get("sender_id", ""),
                    "chunk_index": i,
                    "scope_id": get_scope_id(scope),
                })
        
        if not texts:
            return {
                "success": False,
                "error": "No indexable text content"
            }
        
        if embedding_model is None:
            return {
                "success": False,
                "error": "Embedding model not initialized"
            }
        
        embeddings_list = await embedding_model.aembed_documents(texts)
        
        scope_id = get_scope_id(scope)
        collection_name = f"telegram_{scope_id}"
        
        # Delete existing collection if present
        try:
            existing_collection = chroma_client.get_collection(name=collection_name)
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
        
        # Create new collection
        try:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"scope": json.dumps(scope)}
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                try:
                    chroma_client.delete_collection(name=collection_name)
                    collection = chroma_client.create_collection(
                        name=collection_name,
                        metadata={"scope": json.dumps(scope)}
                    )
                except:
                    collection = chroma_client.get_collection(name=collection_name)
                    collection.delete()
                    collection = chroma_client.create_collection(
                        name=collection_name,
                        metadata={"scope": json.dumps(scope)}
                    )
            else:
                raise
        
        ids = [f"{scope_id}_{i}" for i in range(len(texts))]
        collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "success": True,
            "chunks": len(texts)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to build index: {str(e)}"
        }


@mcp.tool(name="evidence_retrieve")
async def evidence_retrieve(
    k: int,
    channel: str,
    from_date: str,
    to_date: str
) -> Dict[str, Any]:
    """
    Retrieve representative evidence chunks from vector index for summary generation.
    
    Args:
    - k: Number of evidence chunks to return
    - channel: Channel identifier
    - from_date: Start date
    - to_date: End date
    """
    try:
        scope = {
            "channel": channel,
            "from": from_date,
            "to": to_date
        }
        scope_id = get_scope_id(scope)
        collection_name = f"telegram_{scope_id}"
        
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            return {
                "success": False,
                "error": f"Index not found, run build_index first"
            }
        
        # Use generic query for representative evidence
        query_text = "important information key content main topics"
        
        if embedding_model is None:
            return {
                "success": False,
                "error": "Embedding model not initialized"
            }
        
        query_embedding = await embedding_model.aembed_query(query_text)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, 50),
            include=["documents", "metadatas", "distances"]
        )
        
        evidences = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                evidences.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0
                })
        
        return {
            "success": True,
            "evidences": evidences
        }
        
    except Exception as e:
        import traceback
        error_detail = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ evidence_retrieve error: {error_detail}")
        print(f"Traceback:\n{error_trace}")
        return {
            "success": False,
            "error": f"Failed to retrieve evidence: {error_detail}"
        }


@mcp.tool(name="retrieve")
async def retrieve(
    query: str,
    channel: str,
    from_date: str,
    to_date: str
) -> Dict[str, Any]:
    """
    Perform vector retrieval for user queries within specified analysis scope.
    
    Args:
    - query: User query
    - channel: Channel identifier
    - from_date: Start date
    - to_date: End date
    """
    try:
        if not query:
            return {
                "success": False,
                "error": "Query string cannot be empty"
            }
        
        scope = {
            "channel": channel,
            "from": from_date,
            "to": to_date
        }
        scope_id = get_scope_id(scope)
        collection_name = f"telegram_{scope_id}"
        
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            return {
                "success": False,
                "error": f"Index not found, run build_index first"
            }
        
        if embedding_model is None:
            return {
                "success": False,
                "error": "Embedding model not initialized"
            }
        
        query_embedding = await embedding_model.aembed_query(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_results = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                retrieved_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0
                })
        
        return {
            "success": True,
            "results": retrieved_results
        }
        
    except Exception as e:
        import traceback
        error_detail = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ retrieve error: {error_detail}")
        print(f"Traceback:\n{error_trace}")
        return {
            "success": False,
            "error": f"Retrieval failed: {error_detail}"
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Starting MCP server...")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print(f"Index directory: {INDEXES_DIR.absolute()}")
    print(f"Server address: http://0.0.0.0:22331/mcp")
    print(f"Local access: http://127.0.0.1:22331/mcp")
    print("=" * 60)
    
    if embedding_model is None:
        print("\n⚠️  Warning: Embedding model not initialized, retrieval may not work")
        print("   Set OPENAI_API_KEY or install sentence-transformers\n")
    
    print("\nPress Ctrl+C to stop server\n")
    
    mcp.run(transport="http", host="0.0.0.0", port=22331)

