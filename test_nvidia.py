import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

async def main():
    api_key = os.getenv("NVIDIA_API_KEY", "")
    print(f"API Key present: {bool(api_key)}")
    print(f"API Key length: {len(api_key)}")
    
    llm = ChatOpenAI(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        max_tokens=1000,
        temperature=0.6,
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Write a JSON object with a single key 'hello' and value 'world'."),
    ]
    
    print("Sending request (streaming)...")
    try:
        full_text = ""
        async for chunk in llm.astream(messages):
            if chunk.content:
                full_text += chunk.content
                print(chunk.content, end="", flush=True)
        print(f"\n\nTotal Response length: {len(full_text)}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
