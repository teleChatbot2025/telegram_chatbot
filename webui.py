import os
import sys
import time
import signal
import socket
import subprocess
import asyncio
from pathlib import Path
from ui.layout import build_ui
from dotenv import load_dotenv

load_dotenv()

# MCP server config
MCP_SERVER_PORT = 22331
MCP_SERVER_URL = f"http://127.0.0.1:{MCP_SERVER_PORT}/mcp"
MCP_SERVER_SCRIPT = Path(__file__).parent / "mcp_server.py"

mcp_process = None


def check_port_available(host: str, port: int) -> bool:
    """Check if port is available (not in use)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # 0 means connection succeeded (port in use)
    except Exception:
        return True


def wait_for_server(port: int, timeout: int = 45) -> bool:
    """Wait for server to start (check port and TCP connection)."""
    start_time = time.time()
    print(f"   Waiting for server (max {timeout}s)...")
    
    consecutive_success = 0
    
    while time.time() - start_time < timeout:
        if not check_port_available("127.0.0.1", port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()
                
                if result == 0:  # TCP connection succeeded
                    consecutive_success += 1
                    if consecutive_success >= 2:
                        elapsed = int(time.time() - start_time)
                        print(f"   ✅ Server ready (took {elapsed}s)")
                        return True
                else:
                    consecutive_success = 0
            except Exception:
                consecutive_success = 0
        else:
            consecutive_success = 0
        
        elapsed = int(time.time() - start_time)
        if elapsed % 5 == 0 and elapsed > 0:
            print(f"   Waiting... ({elapsed}/{timeout}s)")
        
        time.sleep(1)
    
    return False


def start_mcp_server() -> subprocess.Popen:
    """Start MCP server."""
    global mcp_process
    
    if not MCP_SERVER_SCRIPT.exists():
        print(f"❌ Error: MCP server script not found: {MCP_SERVER_SCRIPT}")
        sys.exit(1)
    
    if not check_port_available("127.0.0.1", MCP_SERVER_PORT):
        print(f"⚠️  Port {MCP_SERVER_PORT} is in use, assuming MCP server is running")
        print(f"   If connection fails, start manually: python {MCP_SERVER_SCRIPT}")
        return None
    
    print("=" * 60)
    print("Starting MCP server...")
    print("=" * 60)
    
    try:
        mcp_process = subprocess.Popen(
            [sys.executable, str(MCP_SERVER_SCRIPT)],
        )
        
        print(f"✅ MCP server process started (PID: {mcp_process.pid})")
        
        if wait_for_server(MCP_SERVER_PORT, timeout=45):
            print(f"✅ MCP server ready: {MCP_SERVER_URL}")
            print("=" * 60)
            return mcp_process
        else:
            print(f"\n❌ MCP server startup timeout (waited 45s)")
            print(f"   Possible reasons:")
            print(f"   1. First startup needs to download models (sentence-transformers)")
            print(f"   2. Server startup error")
            print(f"\n   Suggestions:")
            print(f"   - Start MCP server manually to see logs: python {MCP_SERVER_SCRIPT}")
            print(f"   - Or increase timeout and retry")
            
            if mcp_process:
                try:
                    stdout, stderr = mcp_process.communicate(timeout=1)
                    if stderr:
                        print(f"\n   Server error output:")
                        print(f"   {stderr[:500]}")
                except:
                    pass
                mcp_process.terminate()
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}")
        print(f"   Start manually: python {MCP_SERVER_SCRIPT}")
        sys.exit(1)


def cleanup_mcp_server():
    """Clean up MCP server process."""
    global mcp_process
    if mcp_process:
        print("\nClosing MCP server...")
        try:
            mcp_process.terminate()
            mcp_process.wait(timeout=5)
            print("✅ MCP server closed")
        except subprocess.TimeoutExpired:
            print("⚠️  MCP server not responding, forcing shutdown...")
            mcp_process.kill()
            mcp_process.wait()
        except Exception as e:
            print(f"⚠️  Error closing MCP server: {e}")


def signal_handler(signum, frame):
    """Handle exit signal."""
    cleanup_mcp_server()
    sys.exit(0)


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_mcp_server()
    
    print("\n" + "=" * 60)
    print("Starting Web UI...")
    print("=" * 60)
    
    try:
        demo = build_ui()
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=22337,
            auth_message="Telegram Chat Analyzer",
            show_api=False
        )
    except KeyboardInterrupt:
        print("\nInterrupt signal received...")
    finally:
        cleanup_mcp_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        cleanup_mcp_server()
