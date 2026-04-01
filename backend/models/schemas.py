"""Pydantic models and LangGraph state definition for the multi-agent pipeline."""

from __future__ import annotations
from typing import Literal, TypedDict
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Agent 1 output
# ─────────────────────────────────────────────
class ContextSchema(BaseModel):
    # What the bot is about
    domain: str                              # e.g. "debt collection for telecom"
    persona: str                             # personality and voice of the bot
    tone: str = ""                           # emotional register (e.g. "empathetic but firm")
    summary: str = ""                        # 2-3 sentence overview of the bot's purpose

    # How to write prompts (learned from sample prompts)
    format_rules: list[str]                  # concrete formatting instructions discovered from examples
    style_patterns: list[str]                # structural patterns: how the user opens, frames, constrains
    raw_examples: list[str]                  # verbatim sample prompts — the gold standard reference

    # Behavioral rules that EVERY state prompt must respect
    guardrails: list[str] = []               # things the bot must NEVER do
    escalation_triggers: list[str] = []      # conditions that trigger human handoff
    fallback_behavior: str = ""              # what the bot does on unrecognized input
    error_handling: list[str] = []           # how to handle failures, invalid data, edge cases

    # Data the bot needs to work with
    required_data: list[str] = []            # data the bot must collect (e.g. account_number)
    confirmation_style: str = ""             # how collected data gets confirmed back to user

    # Flow-level rules
    transition_rules: list[str] = []         # rules about how states connect and flow

# ─────────────────────────────────────────────
# Variable Extractor output
# ─────────────────────────────────────────────
class VariableSchema(BaseModel):
    name: str
    description: str
    type: str


# ─────────────────────────────────────────────
# Advanced Retrieval Context
# ─────────────────────────────────────────────
class RetrievalContext(BaseModel):
    """Envelope returned by retrieval_engine.smart_retrieve()."""
    examples: list[str] = []
    scores: list[float] = []
    retrieval_note: str = ""
    is_cold_start: bool = False


# ─────────────────────────────────────────────
# Pattern Abstractor output (Agent 6)
# ─────────────────────────────────────────────
class PatternAnalysis(BaseModel):
    template_skeleton: str = ""
    core_rules: list[str] = []
    anti_patterns: list[str] = []
    slot_priority: list[str] = []


# ─────────────────────────────────────────────
# Per-dimension critic scoring
# ─────────────────────────────────────────────
class DimensionScore(BaseModel):
    persona_consistency: int = 0
    edge_case_coverage: int = 0
    tone_alignment: int = 0
    transition_accuracy: int = 0
    character_limit_compliance: int = 0


class CriticScorecard(BaseModel):
    stage: str = ""
    dimensions: DimensionScore = DimensionScore()
    total: int = 0
    passed: bool = True
    failed_dimensions: list[str] = []
    targeted_instructions: dict[str, str] = {}



# ─────────────────────────────────────────────
# Agent 2 output (one per state)
# ─────────────────────────────────────────────
class StateSpec(BaseModel):
    state_name: str
    intent: str
    expected_user_input: str
    expected_bot_output: str
    dependencies: list[str]
    tags: list[str]


# ─────────────────────────────────────────────
# KB record (ChromaDB + SQLite)
# ─────────────────────────────────────────────
class KBRecord(BaseModel):
    id: str
    state_name: str
    prompt: str
    context_domain: str
    state_intent: str
    tags: list[str]
    source: Literal["seed", "generated", "edited"]
    approved_by: str
    timestamp: str
    run_id: str


# ─────────────────────────────────────────────
# Pipeline draft + result
# ─────────────────────────────────────────────
class PromptDraft(BaseModel):
    state_name: str
    prompt: str
    retrieved_examples: list[str] = []
    status: Literal["pending", "approved", "edited", "discarded"] = "pending"
    edit_content: str | None = None
    discard_reason: str | None = None


class ReviewFinding(BaseModel):
    affected_states: list[str]
    issue_type: str
    description: str
    suggestion: str


class ReviewResult(BaseModel):
    findings: list[ReviewFinding] = []
    overall_note: str = ""


class RunResult(BaseModel):
    run_id: str
    context: ContextSchema
    states: list[StateSpec]
    variables: list[VariableSchema] | None = None
    drafts: list[PromptDraft]
    review_notes: str


# ─────────────────────────────────────────────
# LangGraph pipeline state
# ─────────────────────────────────────────────
class PipelineState(TypedDict, total=False):
    run_id: str
    context_doc: str
    raw_text: str                         # original context doc text for Agent 2/5
    past_prompts: str | None
    state_names: list[str]
    context_schema: dict | None           # serialised ContextSchema
    extracted_variables: list[dict] | None# serialised list[VariableSchema]
    pattern_analysis: dict | None         # serialised PatternAnalysis
    state_specs: list[dict]               # serialised list[StateSpec]
    current_state_index: int
    drafts: list[dict]                    # serialised list[PromptDraft]
    retrieval_contexts: list[dict]        # per-draft RetrievalContext
    review_notes: str | None
    review_findings: list[dict]
    critic_scorecards: list[dict]         # per-stage CriticScorecard
    progress: str
    error: str | None
    # Cold start tracking
    is_cold_start: bool
    cold_start_domains: list[str]
    # Regeneration fields
    regen_state_name: str | None
    regen_reason: str | None


# ─────────────────────────────────────────────
# API request / response models
# ─────────────────────────────────────────────
class RunRequest(BaseModel):
    context_doc: str
    state_names: list[str]
    past_prompts: str | None = None


class RunStatusResponse(BaseModel):
    status: Literal["running", "complete", "error"]
    progress: str | None = None
    result: RunResult | None = None
    error: str | None = None


class ApproveRequest(BaseModel):
    run_id: str
    state_name: str
    edited_prompt: str | None = None


class DiscardRequest(BaseModel):
    run_id: str
    state_name: str
    reason: str = ""
    regenerate: bool = False


class KBListResponse(BaseModel):
    records: list[KBRecord]
    total: int


class ManualKBEntry(BaseModel):
    domain: str
    state_name: str
    intent: str
    prompt: str
    tags: list[str] = []


class AutoKBRequest(BaseModel):
    prompt: str
