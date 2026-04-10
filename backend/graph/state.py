"""Initial state factory for the LangGraph pipeline — case-based + DNA version."""

from backend.models.schemas import PipelineState


def make_initial_state(
    run_id: str,
    context_doc: str,
    state_names: list[str],
    past_prompts: str | None = None,
) -> PipelineState:
    """Create the starting state dict for a pipeline run."""
    return PipelineState(
        run_id=run_id,
        context_doc=context_doc,
        raw_text=context_doc,
        past_prompts=past_prompts,
        state_names=state_names,
        context_schema=None,
        # Case-based pipeline fields
        case_learning_contexts=[],
        state_decompositions=[],
        prioritised_cases=[],
        case_handlers=[],
        extracted_variables=None,
        # Prompt DNA
        mixed_dna=None,
        # Output fields
        state_specs=[],
        current_state_index=0,
        drafts=[],
        retrieval_contexts=[],
        review_notes=None,
        review_findings=[],
        critic_scorecards=[],
        progress="initialised",
        error=None,
        is_cold_start=False,
        cold_start_domains=[],
        regen_state_name=None,
        regen_reason=None,
    )

