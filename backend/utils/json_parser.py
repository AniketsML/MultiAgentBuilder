"""Shared LLM response parsing utilities for all agents."""

import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json(raw_text: str) -> dict | list:
    """Robustly extract JSON (object or array) from an LLM response.

    Handles:
    - <think>...</think> blocks (NVIDIA Nemotron reasoning)
    - Markdown code fences (```json ... ```)
    - Preamble/postamble text around JSON
    """
    text = raw_text.strip()

    # 1. Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Extract from markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # 3. Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Find JSON array [ ... ]
    arr_match = re.search(r'\[.*\]', text, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass

    # 5. Find JSON object { ... }
    obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    # 6. Nothing worked
    logger.error(f"JSON extraction failed. Text (first 500 chars): {text[:500]}")
    logger.error(f"Original (first 300 chars): {raw_text[:300]}")
    raise ValueError(
        f"Could not parse LLM response as JSON. Preview: {text[:200]}"
    )
