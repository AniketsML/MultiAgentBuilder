"""Context Chunker — Semantic windowing for large documents.

Updated for case-based pipeline: new agent categories for KB Case Learner,
Case Prioritiser, Case Writer, and Prompt Assembler.
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
        "agents": ["agent0", "agent2", "case_writer", "assembler"],
    },
    "persona_and_tone": {
        "keywords": [
            "persona", "tone", "voice", "personality", "style", "friendly",
            "formal", "casual", "empathetic", "firm", "warm", "professional",
            "character", "attitude", "manner", "approach", "behavior",
            "name is", "role is", "act as", "you are",
        ],
        "agents": ["agent1", "kb_learner"],
    },
    "guardrails_and_rules": {
        "keywords": [
            "never", "must not", "do not", "don't", "forbidden", "prohibited",
            "guardrail", "rule", "constraint", "limit", "restrict", "compliance",
            "policy", "regulation", "legal", "privacy", "sensitive", "error",
            "fallback", "fail", "invalid", "edge case", "exception",
        ],
        "agents": ["agent1", "case_prioritiser"],
    },
    "data_and_variables": {
        "keywords": [
            "collect", "data", "field", "variable", "input", "number",
            "name", "email", "phone", "address", "amount", "date", "account",
            "payment", "api", "system", "database", "record", "information",
            "capture", "store", "retrieve", "look up",
        ],
        "agents": ["agent2", "case_writer"],
    },
    "cases_and_edge_cases": {
        "keywords": [
            "if the user", "when the user", "objection", "refuses", "angry",
            "escalat", "invalid", "incorrect", "wrong", "repeat", "retry",
            "silent", "no response", "out of scope", "unrelated", "exit",
            "stop", "cancel", "multiple", "ambiguous",
        ],
        "agents": ["kb_learner", "agent2", "case_prioritiser"],
    },
}

# Agents and which categories they need
AGENT_CATEGORIES = {
    "agent0": ["states_and_transitions"],
    "agent1": ["persona_and_tone", "guardrails_and_rules"],
    "agent2": ["states_and_transitions", "data_and_variables", "cases_and_edge_cases"],
    "kb_learner": ["persona_and_tone", "cases_and_edge_cases"],
    "case_prioritiser": ["guardrails_and_rules", "cases_and_edge_cases"],
    "case_writer": ["states_and_transitions", "data_and_variables"],
    "assembler": ["states_and_transitions"],
}

CHUNK_THRESHOLD = 4000  # Docs at or below this pass through raw


def _split_into_chunks(text: str) -> list[str]:
    """Split document by semantic boundaries."""
    if "---" in text:
        parts = re.split(r"\n-{3,}\n", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

    heading_parts = re.split(r"\n(?=#{1,3}\s)", text)
    heading_parts = [p.strip() for p in heading_parts if p.strip()]
    if len(heading_parts) > 1:
        return heading_parts

    parts = re.split(r"\n\n+", text)
    parts = [p.strip() for p in parts if p.strip()]

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

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "states_and_transitions"


def chunk_for_agent(text: str, agent_id: str, max_chars: int = 3000) -> str:
    """Return the relevant portion of a document for a specific agent."""
    if len(text) <= CHUNK_THRESHOLD:
        return text

    chunks = _split_into_chunks(text)
    needed_categories = AGENT_CATEGORIES.get(agent_id, ["states_and_transitions"])

    relevant_chunks = []
    for chunk in chunks:
        category = _categorize_chunk(chunk)
        if category in needed_categories:
            relevant_chunks.append((chunk, category))

    if not relevant_chunks:
        fallback = "\n\n".join(chunks[:3])
        return fallback[:max_chars]

    result = ""
    for chunk, _ in relevant_chunks:
        if len(result) + len(chunk) + 2 > max_chars:
            break
        result = f"{result}\n\n{chunk}" if result else chunk

    if not result:
        result = relevant_chunks[0][0][:max_chars]

    logger.info(
        f"[Chunker] {agent_id}: {len(text)} chars -> {len(result)} chars "
        f"({len(relevant_chunks)}/{len(chunks)} chunks relevant)"
    )
    return result
