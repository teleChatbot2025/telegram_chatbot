"""
Telegram authentication script

Run this script before first use to complete Telegram login.

Usage:
1. Set environment variables:
   - TELEGRAM_API_ID
   - TELEGRAM_API_HASH
   - TELEGRAM_PHONE (optional)

2. Run this script:
   python scripts/auth_telegram.py

3. Enter verification code and password (if 2FA enabled)
"""

import os
import sys
from pathlib import Path

# Add project root to path   
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

# Load environment variables
load_dotenv()

async def main():
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    phone = os.getenv("TELEGRAM_PHONE")
    
    if not api_id or not api_hash:
        print("❌ Error: Missing Telegram API credentials")
        print("\nSet environment variables:")
        print("  - TELEGRAM_API_ID")
        print("  - TELEGRAM_API_HASH")
        print("\nGet them at: https://my.telegram.org/apps")
        return
    
    if not phone:
        phone = input("Enter your Telegram phone number (with country code, e.g. +8613800138000): ")
    
    session_name = "telegram_session"
    client = TelegramClient(session_name, int(api_id), api_hash)
    
    try:
        print("\nConnecting to Telegram...")
        await client.connect()
        
        if not await client.is_user_authorized():
            print(f"\nSending verification code to {phone}...")
            await client.send_code_request(phone)
            
            code = input("Enter verification code: ")
            try:
                await client.sign_in(phone, code)
                print("✅ Login successful!")
            except SessionPasswordNeededError:
                password = input("Enter 2FA password: ")
                await client.sign_in(password=password)
                print("✅ Login successful!")
        else:
            print("✅ Already authorized, no need to login again")
        
        me = await client.get_me()
        print(f"\nLogged in as: {me.first_name} (@{me.username or 'N/A'})")
        print("\n✅ Authentication complete! You can now use MCP server to fetch Telegram messages.")
        
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        sys.exit(1)
    finally:
        await client.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
