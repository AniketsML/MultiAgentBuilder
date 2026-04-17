"""Pydantic models and LangGraph state definition for the multi-agent pipeline.

Redesigned for case-based prompt generation with Prompt DNA:
- Multi-Paradigm Analyzer extracts 8 paradigms of DNA at ingestion time
- Paradigm Mixer selects + blends best DNA per paradigm across KB
- KB Case Learner extracts structured case knowledge from KB
- State Decomposer breaks states into exhaustive case maps
- Case Prioritiser ranks cases by probability x criticality
- Case Writer writes per-case handlers with MixedDNA constraints
- Prompt Assembler composes final prompts with paradigm coherence
"""

from __future__ import annotations
from typing import Any, Literal, TypedDict
import json
from pydantic import BaseModel, field_validator


# ─────────────────────────────────────────────
# Agent 1 output
# ─────────────────────────────────────────────
def _to_str(v: Any) -> str:
    """Coerce any LLM output (str/dict/list/None) to a plain string."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _to_list(v: Any) -> list:
    """Coerce any LLM output to a flat list of strings."""
    if v is None:
        return []
    if isinstance(v, list):
        return [_to_str(i) if not isinstance(i, str) else i for i in v]
    if isinstance(v, dict):
        return [f"{k}: {_to_str(val)}" for k, val in v.items()]
    return [str(v)]


class ContextSchema(BaseModel):
    # What the bot is about
    domain: str = ""                         # e.g. "debt collection for telecom"
    persona: str = ""                        # personality and voice of the bot
    tone: str = ""                           # emotional register (e.g. "empathetic but firm")
    summary: str = ""                        # 2-3 sentence overview of the bot's purpose

    # How to write prompts (learned from sample prompts)
    format_rules: list[str] = []             # concrete formatting instructions discovered from examples
    style_patterns: list[str] = []           # structural patterns: how the user opens, frames, constrains
    raw_examples: list[str] = []             # verbatim sample prompts -- the gold standard reference

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

    # ── Coerce str fields ──────────────────────────────────────────
    @field_validator('domain', 'persona', 'tone', 'summary',
                     'fallback_behavior', 'confirmation_style', mode='before')
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return _to_str(v)

    # ── Coerce list fields ─────────────────────────────────────────
    @field_validator('format_rules', 'style_patterns', 'raw_examples',
                     'guardrails', 'escalation_triggers', 'error_handling',
                     'required_data', 'transition_rules', mode='before')
    @classmethod
    def coerce_list(cls, v: Any) -> list:
        return _to_list(v)


# ─────────────────────────────────────────────
# Variable Schema (now extracted by Agent 2)
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
# KB Case Learning output (replaces PatternAnalysis)
# ─────────────────────────────────────────────
class CaseKnowledge(BaseModel):
    """One case-handling strategy extracted from a KB prompt."""
    case_category: Any                       # e.g. "user_refuses", "wrong_format"
    handling_strategy: Any                   # how the KB prompt handles this case
    variables_used: list[Any] = []           # variables referenced in this case
    transition_target: Any = ""              # where it transitions after this case
    tone_approach: Any = ""                  # tone used for this case type


class CaseLearningContext(BaseModel):
    """Per-state output from KB Case Learner -- parsed knowledge from KB prompts."""
    state_name: Any
    source_count: Any = 0                    # how many KB prompts were analysed
    learned_cases: list[Any] = []            # structured case knowledge
    common_variables: list[Any] = []         # variables seen across multiple KB prompts
    anti_patterns: list[Any] = []            # what KB prompts avoid
    retrieval_note: Any = ""
    is_cold_start: bool = False


# ─────────────────────────────────────────────
# State Decomposer output (upgraded Agent 2)
# ─────────────────────────────────────────────
class CaseSpec(BaseModel):
    """One case within a state decomposition."""
    case_name: Any                           # e.g. "happy_path", "user_refuses"
    category: Any                            # canonical category from taxonomy
    description: Any                         # what triggers this case
    handling_hint: Any                       # suggested handling strategy
    required_variables: list[Any] = []       # variables needed for this case
    transition_to: Any = ""                  # next state after this case
    tone_guidance: Any = ""                  # tone appropriate for this case type


class StateDecomposition(BaseModel):
    """Full decomposition of one state into cases + variables."""
    state_name: Any
    intent: Any                              # one sentence: what this state achieves
    cases: list[Any] = []                    # exhaustive case map
    extracted_variables: list[Any] = []      # variables emerging from cases
    dependencies: list[Any] = []             # other states that must occur first
    tags: list[Any] = []


# ─────────────────────────────────────────────
# Case Prioritiser output
# ─────────────────────────────────────────────
class PrioritisedCase(BaseModel):
    """A case with priority scores."""
    case_name: Any
    category: Any
    description: Any
    handling_hint: Any
    required_variables: list[Any] = []
    transition_to: Any = ""
    tone_guidance: Any = ""
    occurrence_probability: float = 50       # 0-100: how often will this happen
    criticality: float = 50                  # 0-100: cost of handling this badly
    priority_score: float = 50               # combined score
    action: Any = "keep"                     # keep / merge / filter
    merge_into: Any = ""                     # if action=merge, which case absorbs it


class PrioritisedCaseList(BaseModel):
    """Prioritised and filtered case list for one state."""
    state_name: Any
    intent: Any
    cases: list[Any] = []                    # ordered by priority_score desc
    filtered_count: Any = 0                  # how many cases were removed
    merged_count: Any = 0                    # how many cases were merged
    total_char_budget: Any = 4500            # char limit for assembled prompt
    extracted_variables: list[Any] = []
    dependencies: list[Any] = []
    tags: list[Any] = []


# ─────────────────────────────────────────────
# Case Writer output
# ─────────────────────────────────────────────
class CaseHandler(BaseModel):
    """A self-contained handling block for one case."""
    case_name: Any
    category: Any
    condition: Any                           # what triggers this handler
    bot_response: Any                        # what the bot says/does
    variables_used: list[Any] = []
    transition_to: Any = ""
    tone: Any = ""                           # tone used in this handler
    char_count: Any = 0                      # character count of this handler


class CaseWriterOutput(BaseModel):
    """All case handlers for one state."""
    state_name: Any
    handlers: list[Any] = []
    total_char_count: Any = 0


# ─────────────────────────────────────────────
# Per-dimension critic scoring (updated dimensions)
# ─────────────────────────────────────────────
class DimensionScore(BaseModel):
    """Scores vary by gate -- not all dimensions apply to every gate."""
    persona_consistency: float = 0
    edge_case_coverage: float = 0
    tone_alignment: float = 0
    transition_accuracy: float = 0
    character_limit_compliance: float = 0


class CriticScorecard(BaseModel):
    stage: Any = ""
    gate_type: Any = ""                      # which specialized rubric was used
    dimensions: DimensionScore = DimensionScore()
    total: float = 0
    passed: bool = True
    failed_dimensions: list[Any] = []
    targeted_instructions: dict = {}
    root_cause_agent: Any = ""               # if failure caused by upstream agent
    per_case_scores: dict = {}               # per-case-handler quality (after case_writer)


# ─────────────────────────────────────────────
# Legacy StateSpec (kept for backward compatibility in serialization)
# ─────────────────────────────────────────────
class StateSpec(BaseModel):
    state_name: str
    intent: str
    expected_user_input: str
    expected_bot_output: str
    dependencies: list[str]
    tags: list[str]


# ─────────────────────────────────────────────
# KB record (ChromaDB + SQLite) -- now with case metadata
# ─────────────────────────────────────────────
class KBRecord(BaseModel):
    id: str
    state_name: str
    prompt: str
    context_domain: str
    state_intent: str
    tags: list[str]
    source: Literal["seed", "generated", "edited", "approved"]
    approved_by: str
    timestamp: str
    run_id: str
    # New: structured case metadata
    cases_handled: list[str] = []            # case categories handled in this prompt
    case_handling_map: dict[str, str] = {}   # case_category -> handling strategy summary
    variables_used: list[str] = []           # variables referenced in prompt
    transitions: dict[str, str] = {}         # case_category -> next state name


# ─────────────────────────────────────────────
# Pipeline draft + result
# ─────────────────────────────────────────────
class PromptDraft(BaseModel):
    state_name: str
    prompt: str
    retrieved_examples: list[str] = []
    case_breakdown: list[str] = []           # case names handled in this prompt
    status: Literal["pending", "approved", "edited", "discarded"] = "pending"
    edit_content: str | None = None
    discard_reason: str | None = None


class ReviewFinding(BaseModel):
    affected_states: list[Any] = []
    issue_type: Any = ""
    description: Any = ""
    suggestion: Any = ""


class ReviewResult(BaseModel):
    findings: list[Any] = []
    overall_note: Any = ""


class RunResult(BaseModel):
    run_id: str
    context: ContextSchema
    states: list[StateDecomposition]
    variables: list[VariableSchema] | None = None
    drafts: list[PromptDraft]
    review_notes: str
    case_learning_contexts: list[dict] = []  # per-state CaseLearningContext
    prioritised_cases: list[dict] = []       # per-state PrioritisedCaseList
    case_handlers: list[dict] = []           # per-state CaseWriterOutput
    mixed_dna: dict | None = None            # serialised MixedDNA


# ─────────────────────────────────────────────
# Prompt DNA — 8 paradigm extraction system
# ─────────────────────────────────────────────
PARADIGM_NAMES = [
    "structural", "linguistic", "behavioral", "persona",
    "transition", "constraint", "recovery", "rhythm",
]


class ParadigmPrinciples(BaseModel):
    """Extracted principles for one paradigm from one source prompt."""
    paradigm: str                            # one of PARADIGM_NAMES
    principles: list[str] = []               # domain-neutral abstract principles
    confidence: float = 0.0                  # 0.0-1.0: analyzer self-assessed certainty


class PromptDNA(BaseModel):
    """Complete DNA extraction from one KB prompt — all 8 paradigms."""
    source_prompt_id: str                    # KBRecord.id
    domain: str
    use_case: str = ""                       # e.g. "policy_renewal_outbound"
    structural_dna: ParadigmPrinciples = ParadigmPrinciples(paradigm="structural")
    linguistic_dna: ParadigmPrinciples = ParadigmPrinciples(paradigm="linguistic")
    behavioral_dna: ParadigmPrinciples = ParadigmPrinciples(paradigm="behavioral")
    persona_dna: ParadigmPrinciples = ParadigmPrinciples(paradigm="persona")
    transition_dna: ParadigmPrinciples = ParadigmPrinciples(paradigm="transition")
    constraint_dna: ParadigmPrinciples = ParadigmPrinciples(paradigm="constraint")
    recovery_dna: ParadigmPrinciples = ParadigmPrinciples(paradigm="recovery")
    rhythm_dna: ParadigmPrinciples = ParadigmPrinciples(paradigm="rhythm")
    timestamp: str = ""

    def get_paradigm(self, name: str) -> ParadigmPrinciples:
        return getattr(self, f"{name}_dna", ParadigmPrinciples(paradigm=name))

    def all_paradigms(self) -> list[ParadigmPrinciples]:
        return [self.get_paradigm(p) for p in PARADIGM_NAMES]


class ParadigmConflict(BaseModel):
    """A conflict between two selected paradigm sources."""
    paradigm_a: str                          # e.g. "structural"
    paradigm_b: str                          # e.g. "rhythm"
    conflict_description: str                # what conflicts
    resolution: str                          # how it was resolved
    priority_winner: str                     # which paradigm took priority


class MixedDNA(BaseModel):
    """Blended DNA from multiple sources — one best-of per paradigm."""
    structural: ParadigmPrinciples = ParadigmPrinciples(paradigm="structural")
    linguistic: ParadigmPrinciples = ParadigmPrinciples(paradigm="linguistic")
    behavioral: ParadigmPrinciples = ParadigmPrinciples(paradigm="behavioral")
    persona: ParadigmPrinciples = ParadigmPrinciples(paradigm="persona")
    transition: ParadigmPrinciples = ParadigmPrinciples(paradigm="transition")
    constraint: ParadigmPrinciples = ParadigmPrinciples(paradigm="constraint")
    recovery: ParadigmPrinciples = ParadigmPrinciples(paradigm="recovery")
    rhythm: ParadigmPrinciples = ParadigmPrinciples(paradigm="rhythm")
    source_map: dict[str, str] = {}          # paradigm -> source_prompt_id
    conflicts: list[ParadigmConflict] = []   # detected and resolved conflicts
    is_cold_start: bool = False              # no DNA available

    def get_paradigm(self, name: str) -> ParadigmPrinciples:
        return getattr(self, name, ParadigmPrinciples(paradigm=name))

    def all_principles_flat(self) -> list[str]:
        """All principles across all paradigms as a flat list."""
        result = []
        for p in PARADIGM_NAMES:
            pp = self.get_paradigm(p)
            for pr in pp.principles:
                result.append(f"[{p.upper()}] {pr}")
        return result


# ─────────────────────────────────────────────
# Agent prompt versioning (for improver regression detection)
# ─────────────────────────────────────────────
class PromptVersion(BaseModel):
    """One version of an agent's system prompt."""
    version: int
    prompt: str
    timestamp: str
    trigger_feedback: str = ""               # what feedback triggered this rewrite
    scores_before: dict[str, int] = {}       # review scores before this version
    scores_after: dict[str, int] = {}        # review scores after this version ran


# ─────────────────────────────────────────────
# LangGraph pipeline state
# ─────────────────────────────────────────────
class PipelineState(TypedDict, total=False):
    run_id: str
    context_doc: str
    raw_text: str                            # original context doc text
    past_prompts: str | None
    state_names: list[str]
    context_schema: dict | None              # serialised ContextSchema

    # Case-based pipeline fields
    case_learning_contexts: list[dict]       # per-state CaseLearningContext
    state_decompositions: list[dict]         # per-state StateDecomposition
    prioritised_cases: list[dict]            # per-state PrioritisedCaseList
    case_handlers: list[dict]               # per-state CaseWriterOutput
    extracted_variables: list[dict] | None   # aggregated from all decompositions

    # Prompt DNA fields
    mixed_dna: dict | None                   # serialised MixedDNA

    # Legacy fields retained for compatibility
    state_specs: list[dict]                  # kept for Agent 4 / finalise
    current_state_index: int
    drafts: list[dict]                       # serialised list[PromptDraft]
    retrieval_contexts: list[dict]           # per-draft RetrievalContext
    review_notes: str | None
    review_findings: list[dict]
    critic_scorecards: list[dict]            # per-stage CriticScorecard
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


class MasterFeedbackRequest(BaseModel):
    """Route qualitative feedback from Master Agent chat to improvers."""
    feedback: str
    affected_agents: list[str] = []          # e.g. ["case_writer", "assembler"]


class MasterActionRequest(BaseModel):
    """Master Agent write actions."""
    action: Literal["approve", "discard", "add_kb", "trigger_improvement", "override_priority"]
    run_id: str = ""
    state_name: str = ""
    payload: dict = {}
