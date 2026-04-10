"""Multi-Paradigm Analyzer — Prompt DNA extraction agent.

Runs at KB INGESTION time (not during the generation pipeline).
Takes a raw prompt and produces a PromptDNA object with all 8 paradigms
extracted as abstract, domain-neutral principles.

Critical constraint: every extracted principle must be stated in domain-neutral
language. "Acknowledge the user's concern before redirecting" — not
"say 'I understand you have questions about your policy'".

Makes 8 separate focused LLM calls — one per paradigm.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json
from backend.models.schemas import PromptDNA, ParadigmPrinciples, PARADIGM_NAMES
from backend.kb import chroma_client
from backend.kb import sqlite_db

logger = logging.getLogger(__name__)


# ──────────── Per-paradigm extraction prompts ────────────

PARADIGM_PROMPTS = {
    "structural": """Analyse the STRUCTURAL DNA of this prompt.

Extract abstract principles about HOW the prompt is architecturally organized:
- What ordering logic exists? (primary flow first, then objections, then errors — or something else?)
- How are case transitions marked? (implicit vs explicit conditional language)
- What's the information density? (tightly specified vs loose guidance)
- How are retry cycles structured?
- Where does escalation sit in the prompt hierarchy?
- How are sections delineated? (headers, numbered lists, conditional blocks, prose?)

STATE EVERY PRINCIPLE IN DOMAIN-NEUTRAL LANGUAGE.
"objection handling always precedes system error handling" — not "policy questions come before system errors".
These principles must be transferable to ANY domain.""",

    "linguistic": """Analyse the LINGUISTIC DNA of this prompt.

Extract abstract principles about the language operating system:
- Sentence length distribution (short declarative vs longer explanatory)
- Vocabulary level (B1 consumer vs B2 professional vs C1 technical)
- Formality spectrum (contractions, colloquialisms, directness)
- Empathy marker patterns (acknowledge-then-redirect vs validate-then-continue)
- Instruction specificity (prescriptive exact wording vs directional guidance)
- How does formality shift within the prompt based on emotional context?

STATE EVERY PRINCIPLE IN DOMAIN-NEUTRAL LANGUAGE.""",

    "behavioral": """Analyse the BEHAVIORAL DNA of this prompt.

Extract abstract principles about the decision philosophy:
- Retry philosophy (how many attempts before escalation, how does each retry differ?)
- Ambiguity handling philosophy (ask clarifying question vs make best guess vs offer options?)
- Refusal handling philosophy (acknowledge → explain → offer alternative vs immediate redirect?)
- Missing information philosophy (ask one field at a time vs ask all missing fields at once?)
- User silence philosophy (prompt once vs prompt twice vs hand off?)
- What is the bot's overall agency model? (proactive vs reactive)

STATE EVERY PRINCIPLE IN DOMAIN-NEUTRAL LANGUAGE.""",

    "persona": """Analyse the PERSONA DNA of this prompt.

Extract abstract principles about how personality is technically implemented:
- Where does persona manifest? (greeting only, every response, or only in recovery?)
- How is warmth expressed technically? (word choice, question framing, affirmation patterns)
- How does the persona modulate under stress? (more formal when user is angry?)
- What persona consistency markers appear across case types?
- Is the persona role-based (agent, assistant, advisor) or personality-based (friendly, professional)?

STATE EVERY PRINCIPLE IN DOMAIN-NEUTRAL LANGUAGE.""",

    "transition": """Analyse the TRANSITION DNA of this prompt.

Extract abstract principles about state change management:
- What triggers a transition? (explicit confirmation vs implicit completion vs timeout?)
- How are transitions communicated to the user? (explicit announcement vs seamless continuation?)
- What confirmation logic exists before transitioning?
- How are conditional transitions handled? (different next states based on outcome?)
- Are transitions summarized? ("Now that we have X, let's move to Y")

STATE EVERY PRINCIPLE IN DOMAIN-NEUTRAL LANGUAGE.""",

    "constraint": """Analyse the CONSTRAINT DNA of this prompt.

Extract abstract principles about the guardrail philosophy:
- What categories of things does the bot explicitly not do?
- How are out-of-scope requests handled? (redirect vs acknowledge-and-redirect vs hard stop?)
- What escalation triggers exist and how sensitive are they?
- How are legal/compliance constraints enforced in language?
- Are prohibitions stated as negatives ("I can't") or redirected to positives ("What I CAN do...")?

STATE EVERY PRINCIPLE IN DOMAIN-NEUTRAL LANGUAGE.""",

    "recovery": """Analyse the RECOVERY DNA of this prompt.

Extract abstract principles about failure handling:
- How does the prompt recover from misunderstanding loops?
- How does tone shift across retry attempts? (static vs escalating patience vs de-escalating formality?)
- What graceful degradation path exists? (reduce scope vs hand off vs offer callback?)
- How are multiple simultaneous failures handled?
- Does each retry reframe the question differently or repeat it?

STATE EVERY PRINCIPLE IN DOMAIN-NEUTRAL LANGUAGE.""",

    "rhythm": """Analyse the RHYTHM DNA of this prompt.

Extract abstract principles about conversational pacing:
- How verbose are bot turns? (brief single-ask vs multi-part explanations)
- How much does the bot lead vs follow the user?
- What listening signals are embedded? (acknowledgment before response)
- How is silence handled?
- Does verbosity shift based on case type? (errors brief, happy path verbose, or vice versa?)
- What's the information-per-turn ratio?

STATE EVERY PRINCIPLE IN DOMAIN-NEUTRAL LANGUAGE.""",
}


BASE_SYSTEM = """You are a Prompt DNA Analyzer — you dissect prompts to extract deep architectural principles.

You will receive a prompt and must extract abstract, domain-neutral principles for one specific paradigm.

RULES:
1. Each principle must be TRANSFERABLE — it must make sense applied to any domain.
2. Each principle must be SPECIFIC — "be empathetic" is useless. "Acknowledge the user's stated concern before any redirect, using first-person language" is useful.
3. Each principle must be OBSERVABLE — you must be able to point to where in the prompt this principle is demonstrated.
4. Limit to 3-7 principles per paradigm. Quality over quantity.
5. Assess your own confidence (0.0-1.0) in the extraction. Low confidence = prompt doesn't clearly demonstrate this paradigm.

Respond with ONLY a JSON object:
{
  "paradigm": "<paradigm_name>",
  "principles": ["principle 1", "principle 2", ...],
  "confidence": 0.85
}"""


async def _extract_single_paradigm(
    prompt_text: str,
    paradigm: str,
    domain: str,
    llm,
) -> ParadigmPrinciples:
    """Extract principles for one paradigm from one prompt."""
    paradigm_instruction = PARADIGM_PROMPTS.get(paradigm, "")

    user_msg = f"""DOMAIN (for context only — principles must be domain-neutral): {domain}

PROMPT TO ANALYSE:
---
{prompt_text[:6000]}
---

{paradigm_instruction}"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=BASE_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        parsed = extract_json(response.content)

        principles = parsed.get("principles", [])
        confidence = float(parsed.get("confidence", 0.5))

        # Validate: filter out domain-specific or too-vague principles
        clean_principles = []
        for p in principles:
            if isinstance(p, str) and len(p) > 10:
                clean_principles.append(p)

        return ParadigmPrinciples(
            paradigm=paradigm,
            principles=clean_principles[:7],  # cap at 7
            confidence=min(max(confidence, 0.0), 1.0),
        )
    except Exception as e:
        logger.warning(f"[DNAAnalyzer] Failed to extract {paradigm}: {e}")
        return ParadigmPrinciples(paradigm=paradigm, principles=[], confidence=0.0)


async def extract_prompt_dna(
    prompt_text: str,
    source_prompt_id: str,
    domain: str,
    use_case: str = "",
) -> PromptDNA:
    """Extract all 8 paradigms from a prompt. Makes 8 parallel LLM calls.

    This is the main entry point called at KB ingestion time.
    """
    llm = get_llm(max_tokens=1500)

    # Run all 8 extractions in parallel
    tasks = [
        _extract_single_paradigm(prompt_text, paradigm, domain, llm)
        for paradigm in PARADIGM_NAMES
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Build the PromptDNA object
    paradigm_map = {}
    for i, paradigm in enumerate(PARADIGM_NAMES):
        result = results[i]
        if isinstance(result, Exception):
            logger.warning(f"[DNAAnalyzer] {paradigm} failed: {result}")
            paradigm_map[paradigm] = ParadigmPrinciples(
                paradigm=paradigm, principles=[], confidence=0.0
            )
        else:
            paradigm_map[paradigm] = result

    dna = PromptDNA(
        source_prompt_id=source_prompt_id,
        domain=domain,
        use_case=use_case,
        structural_dna=paradigm_map.get("structural", ParadigmPrinciples(paradigm="structural")),
        linguistic_dna=paradigm_map.get("linguistic", ParadigmPrinciples(paradigm="linguistic")),
        behavioral_dna=paradigm_map.get("behavioral", ParadigmPrinciples(paradigm="behavioral")),
        persona_dna=paradigm_map.get("persona", ParadigmPrinciples(paradigm="persona")),
        transition_dna=paradigm_map.get("transition", ParadigmPrinciples(paradigm="transition")),
        constraint_dna=paradigm_map.get("constraint", ParadigmPrinciples(paradigm="constraint")),
        recovery_dna=paradigm_map.get("recovery", ParadigmPrinciples(paradigm="recovery")),
        rhythm_dna=paradigm_map.get("rhythm", ParadigmPrinciples(paradigm="rhythm")),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    total_principles = sum(len(p.principles) for p in dna.all_paradigms())
    avg_confidence = (
        sum(p.confidence for p in dna.all_paradigms()) / 8
        if dna.all_paradigms() else 0
    )

    logger.info(
        f"[DNAAnalyzer] Extracted {total_principles} principles across 8 paradigms "
        f"(avg confidence: {avg_confidence:.2f}) from {source_prompt_id}"
    )

    return dna


async def analyze_and_store(
    prompt_text: str,
    source_prompt_id: str,
    domain: str,
    use_case: str = "",
) -> PromptDNA:
    """Extract DNA and persist to both ChromaDB and SQLite.

    Called from kb_writer when a prompt is approved or seeded.
    """
    dna = await extract_prompt_dna(prompt_text, source_prompt_id, domain, use_case)
    dna_dict = dna.model_dump()

    # Store in ChromaDB for semantic querying
    try:
        chroma_client.upsert_dna(dna_dict)
    except Exception as e:
        logger.warning(f"[DNAAnalyzer] ChromaDB DNA store failed: {e}")

    # Store in SQLite for persistent retrieval
    try:
        await sqlite_db.insert_dna_record(dna_dict)
    except Exception as e:
        logger.warning(f"[DNAAnalyzer] SQLite DNA store failed: {e}")

    return dna
