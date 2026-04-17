"""Shared LLM client — NVIDIA Nemotron via OpenAI-compatible API.

Uses a singleton AsyncOpenAI client for connection pool reuse.
Includes timeout, retry-with-backoff, and per-agent temperature support.
"""

import asyncio
import logging
from openai import AsyncOpenAI, APITimeoutError, APIStatusError, APIConnectionError

from backend.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, MODEL_NAME

logger = logging.getLogger(__name__)

# ── Singleton client — shared across all agents ──────────────────────────────
_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
            timeout=120.0,          # 2 min hard timeout per request
            max_retries=0,          # We handle retries ourselves (with backoff)
        )
    return _client


# ── Per-agent recommended temperatures ───────────────────────────────────────
AGENT_TEMPERATURES = {
    "dna_analyzer":     0.2,   # Deterministic principle extraction
    "paradigm_mixer":   0.3,   # Scoring/selection — should be consistent
    "kb_learner":       0.3,   # Factual extraction from KB prompts
    "agent1":           0.4,   # Context analysis — some creativity OK
    "agent2":           0.5,   # Case decomposition
    "case_prioritiser": 0.3,   # Scoring — should be consistent
    "case_writer":      0.6,   # Writing handlers — needs creativity
    "assembler":        0.65,  # Final composition — most creative stage
    "agent4":           0.2,   # Reviewer — deterministic scoring
    "review_agent":     0.2,   # Gate reviewer — deterministic
    "master":           0.7,   # Chat agent — conversational
}

DEFAULT_TEMPERATURE = 0.6
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # seconds, doubles each retry


class LLMWrapper:
    def __init__(self, max_tokens: int, temperature: float = DEFAULT_TEMPERATURE):
        self.client = _get_client()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = MODEL_NAME

    async def ainvoke(self, messages) -> object:
        """Invoke the LLM with retry-on-failure and timeout handling."""

        # Convert LangChain messages to OpenAI format
        oai_messages = []
        for m in messages:
            if m.type == "system":
                role = "system"
            elif m.type == "human":
                role = "user"
            else:
                role = "assistant"
            oai_messages.append({"role": role, "content": m.content})

        logger.info(
            f"[LLM] → {self.model} | msgs={len(oai_messages)} | "
            f"max_tokens={self.max_tokens} | temp={self.temperature}"
        )

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                full_text = await self._stream_completion(oai_messages)
                logger.info(f"[LLM] ← {len(full_text)} chars (attempt {attempt})")
                response = _make_response(full_text)
                return response

            except APITimeoutError as e:
                last_error = e
                logger.warning(f"[LLM] Timeout on attempt {attempt}/{MAX_RETRIES}")

            except APIStatusError as e:
                last_error = e
                if e.status_code == 429:
                    # Rate limit — always retry with longer backoff
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + 5
                    logger.warning(
                        f"[LLM] Rate limited (429), waiting {delay:.0f}s "
                        f"(attempt {attempt}/{MAX_RETRIES})"
                    )
                elif e.status_code >= 500:
                    # Server error — retry
                    logger.warning(
                        f"[LLM] Server error {e.status_code} on attempt {attempt}/{MAX_RETRIES}"
                    )
                else:
                    # 4xx client error (bad request etc.) — don't retry
                    logger.error(f"[LLM] Client error {e.status_code}: {e.message}")
                    raise

            except APIConnectionError as e:
                last_error = e
                logger.warning(f"[LLM] Connection error on attempt {attempt}/{MAX_RETRIES}: {e}")

            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info(f"[LLM] Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        logger.error(f"[LLM] All {MAX_RETRIES} attempts failed. Last error: {last_error}")
        raise last_error

    async def _stream_completion(self, oai_messages: list) -> str:
        """Run a single streaming completion and return the full text."""
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            temperature=self.temperature,
            top_p=0.95,
            max_tokens=self.max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True,
        )

        full_text = ""
        async for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content

        return full_text


def _make_response(content: str) -> object:
    """Create a mock response object matching the LangChain interface."""
    response = type('Response', (), {})()
    response.content = content
    response.response_metadata = {'finish_reason': 'stop'}
    return response


def get_llm(max_tokens: int = 2000, agent_id: str = "") -> LLMWrapper:
    """Return a configured LLM wrapper.

    Args:
        max_tokens: Max tokens for this call.
        agent_id: Optional agent identifier to select the right temperature.
                  Uses DEFAULT_TEMPERATURE if not specified or not in table.
    """
    temperature = AGENT_TEMPERATURES.get(agent_id, DEFAULT_TEMPERATURE)
    return LLMWrapper(max_tokens=max_tokens, temperature=temperature)
