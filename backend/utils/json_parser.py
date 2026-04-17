"""Shared LLM response parsing utilities for all agents."""

import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json(raw_text: str, expected_type: type | None = None) -> dict | list:
    """Robustly extract JSON (object or array) from an LLM response.

    Handles:
    - <think>...</think> blocks (NVIDIA Nemotron reasoning)
    - Markdown code fences (```json ... ```)
    - Preamble/postamble text around JSON
    - Truncated responses (detected and reported clearly)
    - Greedy-regex false positives (uses bracket-balanced scanner)

    Args:
        raw_text: Raw LLM response string.
        expected_type: Optional — dict or list. If provided and the parsed
            result is the wrong type, raises ValueError with a clear message.
    """
    text = raw_text.strip()

    # 1. Remove <think>...</think> blocks (Nemotron reasoning)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Extract from markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # 3. Try direct parse first (cleanest path)
    try:
        result = json.loads(text)
        return _validate_type(result, expected_type, "direct parse")
    except json.JSONDecodeError:
        pass

    # 4. Use bracket-balanced scanner — try whichever bracket appears first in text
    #    so that an array with preamble (e.g. "Sure! Here is the list: [...]")
    #    is found correctly instead of grabbing the first { inside it.
    obj_pos = text.find('{')
    arr_pos = text.find('[')

    if obj_pos == -1 and arr_pos == -1:
        pairs = []
    elif obj_pos == -1:
        pairs = [('[', ']'), ('{', '}')]
    elif arr_pos == -1:
        pairs = [('{', '}'), ('[', ']')]
    elif arr_pos < obj_pos:
        pairs = [('[', ']'), ('{', '}')]
    else:
        pairs = [('{', '}'), ('[', ']')]

    for opener, closer in pairs:
        extracted = _balanced_extract(text, opener, closer)
        if extracted:
            try:
                result = json.loads(extracted)
                return _validate_type(result, expected_type, f"bracket scan ({opener})")
            except json.JSONDecodeError:
                pass

    # 5. Truncation detection — helps diagnose max_tokens cuts
    _check_truncation(raw_text)

    # 6. Nothing worked — log enough context to debug
    logger.error(
        f"[JSONParser] All extraction attempts failed.\n"
        f"  Text preview (500): {text[:500]}\n"
        f"  Original preview (300): {raw_text[:300]}"
    )
    raise ValueError(
        f"Could not parse LLM response as JSON. "
        f"Preview: {text[:200]!r}"
    )


# ──────────────── Helpers ────────────────

def _balanced_extract(text: str, opener: str, closer: str) -> str | None:
    """Find the first balanced JSON block starting with `opener`.

    Uses a depth counter instead of greedy regex, so nested objects/arrays
    don't trip it up and preamble text before the first opener is skipped.
    """
    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None  # Unbalanced — likely truncated


def _validate_type(result: dict | list, expected_type: type | None, source: str) -> dict | list:
    """Validate the parsed result against the expected type if specified."""
    if expected_type is not None and not isinstance(result, expected_type):
        raise ValueError(
            f"[JSONParser] Expected {expected_type.__name__} from {source}, "
            f"got {type(result).__name__}. "
            f"Value preview: {str(result)[:200]}"
        )
    return result


def _check_truncation(raw_text: str):
    """Log a warning if the response looks truncated (no closing bracket found)."""
    stripped = raw_text.strip()
    has_open_obj = '{' in stripped
    has_open_arr = '[' in stripped
    has_close_obj = '}' in stripped
    has_close_arr = ']' in stripped

    if (has_open_obj and not has_close_obj) or (has_open_arr and not has_close_arr):
        logger.warning(
            "[JSONParser] Response appears TRUNCATED — opening bracket found but no closing bracket. "
            "Consider increasing max_tokens for this agent. "
            f"Response length: {len(raw_text)} chars."
        )


async def json_ainvoke_with_retry(llm, messages, expected_type: type | None = None, max_retries: int = 3) -> dict | list:
    """Invoke LLM and parse JSON, retrying on truncation or malformed output."""
    import asyncio
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await llm.ainvoke(messages)
            parsed = extract_json(response.content, expected_type)
            return parsed
        except ValueError as e:
            last_error = e
            logger.warning(f"[JSONParser] Attempt {attempt}/{max_retries} failed to produce valid JSON: {e}")
            if attempt < max_retries:
                # Add a hint to the LLM to fix its output
                messages.append(type('Response', (), {'type': 'assistant', 'content': response.content})())
                messages.append(type('HumanMessage', (), {
                    'type': 'human', 
                    'content': f"Your previous response was invalid or truncated JSON. Please provide the complete, valid JSON object requested. Error: {e}"
                })())
                await asyncio.sleep(2)
                
    logger.error(f"[JSONParser] All {max_retries} attempts failed to produce valid JSON.")
    raise last_error
