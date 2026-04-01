"""Context Chunker — Semantic windowing for large documents.

For docs >4000 chars, splits by semantic coherence and maps chunks
to agent categories so each agent gets a focused context window.
Docs ≤4000 chars pass through raw — no chunking.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Chunk categories and their agent mappings
CHUNK_CATEGORIES = {
    "states_and_transitions": {
        "keywords": [
            "state", "step", "flow", "transition", "handoff", "escalat",
            "transfer", "move to", "proceed", "next step", "conversation flow",
            "greeting", "farewell", "opening", "closing", "sequence",
        ],
        "agents": ["agent0", "agent3", "agent6"],
    },
    "persona_and_tone": {
        "keywords": [
            "persona", "tone", "voice", "personality", "style", "friendly",
            "formal", "casual", "empathetic", "firm", "warm", "professional",
            "character", "attitude", "manner", "approach", "behavior",
            "name is", "role is", "act as", "you are",
        ],
        "agents": ["agent1", "agent6"],
    },
    "guardrails_and_rules": {
        "keywords": [
            "never", "must not", "do not", "don't", "forbidden", "prohibited",
            "guardrail", "rule", "constraint", "limit", "restrict", "compliance",
            "policy", "regulation", "legal", "privacy", "sensitive", "error",
            "fallback", "fail", "invalid", "edge case", "exception",
        ],
        "agents": ["agent1"],
    },
    "data_and_variables": {
        "keywords": [
            "collect", "data", "field", "variable", "input", "number",
            "name", "email", "phone", "address", "amount", "date", "account",
            "payment", "api", "system", "database", "record", "information",
            "capture", "store", "retrieve", "look up",
        ],
        "agents": ["agent3", "agent5"],
    },
}

# Agents and which categories they need
AGENT_CATEGORIES = {
    "agent0": ["states_and_transitions"],
    "agent1": ["persona_and_tone", "guardrails_and_rules"],
    "agent3": ["states_and_transitions", "data_and_variables"],
    "agent5": ["data_and_variables"],
    "agent6": ["persona_and_tone", "states_and_transitions"],
}

CHUNK_THRESHOLD = 4000  # Docs at or below this pass through raw


def _split_into_chunks(text: str) -> list[str]:
    """Split document by semantic boundaries: section headings, ---, or double newlines."""
    # Try --- separator first
    if "---" in text:
        parts = re.split(r"\n-{3,}\n", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

    # Try markdown headings
    heading_parts = re.split(r"\n(?=#{1,3}\s)", text)
    heading_parts = [p.strip() for p in heading_parts if p.strip()]
    if len(heading_parts) > 1:
        return heading_parts

    # Fall back to double newlines
    parts = re.split(r"\n\n+", text)
    parts = [p.strip() for p in parts if p.strip()]

    # If we got too many tiny chunks, merge adjacent ones to get ~500-800 char chunks
    if len(parts) > 10:
        merged = []
        current = ""
        for p in parts:
            if len(current) + len(p) < 800:
                current = f"{current}\n\n{p}" if current else p
            else:
                if current:
                    merged.append(current)
                current = p
        if current:
            merged.append(current)
        return merged

    return parts if parts else [text]


def _categorize_chunk(chunk: str) -> str:
    """Classify a chunk into the best-matching category."""
    chunk_lower = chunk.lower()
    scores = {}

    for category, config in CHUNK_CATEGORIES.items():
        score = sum(1 for kw in config["keywords"] if kw in chunk_lower)
        scores[category] = score

    # Return the highest-scoring category, default to states_and_transitions
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "states_and_transitions"


def chunk_for_agent(text: str, agent_id: str, max_chars: int = 3000) -> str:
    """Return the relevant portion of a document for a specific agent.

    - Docs ≤4000 chars → returns full text (no chunking)
    - Larger docs → chunks semantically, filters by agent's category needs,
      returns concatenated relevant chunks up to max_chars
    """
    if len(text) <= CHUNK_THRESHOLD:
        return text

    chunks = _split_into_chunks(text)
    needed_categories = AGENT_CATEGORIES.get(agent_id, ["states_and_transitions"])

    # Score and filter chunks
    relevant_chunks = []
    for chunk in chunks:
        category = _categorize_chunk(chunk)
        if category in needed_categories:
            relevant_chunks.append((chunk, category))

    # If no relevant chunks found, return first N chunks as fallback
    if not relevant_chunks:
        fallback = "\n\n".join(chunks[:3])
        return fallback[:max_chars]

    # Concatenate relevant chunks up to max_chars
    result = ""
    for chunk, _ in relevant_chunks:
        if len(result) + len(chunk) + 2 > max_chars:
            break
        result = f"{result}\n\n{chunk}" if result else chunk

    if not result:
        result = relevant_chunks[0][0][:max_chars]

    logger.info(
        f"[Chunker] {agent_id}: {len(text)} chars → {len(result)} chars "
        f"({len(relevant_chunks)}/{len(chunks)} chunks relevant)"
    )
    return result
