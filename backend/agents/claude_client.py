"""Shared LLM client — NVIDIA Nemotron via OpenAI-compatible API."""

from openai import AsyncOpenAI
from backend.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, MODEL_NAME

class LLMWrapper:
    def __init__(self, max_tokens: int):
        self.client = AsyncOpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
        )
        self.max_tokens = max_tokens
        self.model = MODEL_NAME

    async def ainvoke(self, messages) -> type('Response', (), {}):
        """Mock the LangChain ainvoke interface using raw OpenAI streaming."""
        
        # Convert LangChain messages to OpenAI format
        oai_messages = []
        for m in messages:
            role = "system" if m.type == "system" else "user" if m.type == "human" else "assistant"
            oai_messages.append({"role": role, "content": m.content})
            
        print(f"[LLM] Sending request to {self.model} ({len(oai_messages)} messages)...")
            
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=self.max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )
        
        full_text = ""
        async for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
                
        print(f"[LLM] Received {len(full_text)} characters")
        
        # Mock the response object expected by agents
        response = type('Response', (), {})()
        response.content = full_text
        response.response_metadata = {'finish_reason': 'stop'}
        return response


def get_llm(max_tokens: int = 2000) -> LLMWrapper:
    """Return a wrapper that mimics ChatOpenAI but uses raw AsyncOpenAI."""
    return LLMWrapper(max_tokens)
