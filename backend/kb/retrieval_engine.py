"""Advanced Retrieval Engine — Multi-signal + BM25 Hybrid + MMR + Cold Start.

Updated for case-based pipeline: supports case-category filtering and
returns structured case metadata alongside prompt text.
"""

import math
import logging
from datetime import datetime, timezone

from rank_bm25 import BM25Okapi

from backend.kb import chroma_client
from backend.kb import sqlite_db
from backend.models.schemas import RetrievalContext

logger = logging.getLogger(__name__)

# ──────────────── BM25 Index Cache ────────────────
_bm25_cache: dict[str, dict] = {}


def invalidate_bm25_cache(domain: str):
    """Called after upsert_to_kb() to force a rebuild on next query."""
    _bm25_cache.pop(domain, None)
    _bm25_cache.pop("__all__", None)


async def _get_or_build_bm25(domain: str | None) -> tuple:
    """Get cached BM25 index or build a new one."""
    cache_key = domain or "__all__"

    if cache_key in _bm25_cache:
        cached = _bm25_cache[cache_key]
        if domain:
            latest_ts = await sqlite_db.get_last_upsert_timestamp(domain)
            if latest_ts == cached.get("timestamp"):
                return cached["index"], cached["docs"]
        else:
            return cached["index"], cached["docs"]

    records = await sqlite_db.get_all_prompts_for_bm25(domain)
    if not records:
        return None, []

    corpus = []
    for r in records:
        # Include cases_handled in BM25 indexing for case-category search
        cases_text = " ".join(r.get("cases_handled", []))
        text = f"{r['prompt']} {r['state_intent']} {cases_text}"
        corpus.append(text.lower().split())

    bm25 = BM25Okapi(corpus)

    latest_ts = None
    if domain:
        latest_ts = await sqlite_db.get_last_upsert_timestamp(domain)

    _bm25_cache[cache_key] = {
        "index": bm25,
        "docs": records,
        "timestamp": latest_ts,
    }

    return bm25, records


# ──────────────── Core Retrieval ────────────────

def _build_where_clause(domain: str, strict: bool = True) -> dict | None:
    if not domain:
        return None
    if strict:
        return {
            "$and": [
                {"context_domain": {"$eq": domain}},
                {"source": {"$in": ["edited", "approved", "seed"]}},
            ]
        }
    else:
        return {"context_domain": {"$eq": domain}}


def _parse_chroma_to_candidates(results: dict) -> list[dict]:
    """Parse raw ChromaDB response into candidate dicts with case metadata."""
    candidates = []
    if not results["ids"] or not results["ids"][0]:
        return candidates

    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        distance = results["distances"][0][i] if results.get("distances") else 1.0
        similarity = max(0.0, 1.0 - distance)

        tags = meta.get("tags", "")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        cases_handled = meta.get("cases_handled", "")
        if isinstance(cases_handled, str):
            cases_handled = [c.strip() for c in cases_handled.split(",") if c.strip()]

        variables_used = meta.get("variables_used", "")
        if isinstance(variables_used, str):
            variables_used = [v.strip() for v in variables_used.split(",") if v.strip()]

        candidates.append({
            "id": doc_id,
            "prompt": doc,
            "similarity": similarity,
            "domain": meta.get("context_domain", ""),
            "state_name": meta.get("state_name", ""),
            "state_intent": meta.get("state_intent", ""),
            "tags": tags,
            "cases_handled": cases_handled,
            "variables_used": variables_used,
            "source": meta.get("source", "generated"),
            "timestamp": meta.get("timestamp", ""),
        })

    return candidates


def _compute_weighted_score(
    candidate: dict,
    target_domain: str,
    target_tags: list[str],
    target_case_categories: list[str] | None = None,
) -> float:
    """Compute weighted score: 0.4*semantic + 0.25*domain + 0.15*tag + 0.2*case_overlap."""
    semantic = candidate["similarity"]

    # Domain match
    domain_score = 0.0
    if candidate["domain"] == target_domain:
        domain_score = 1.0
    elif target_domain and target_domain in candidate["domain"]:
        domain_score = 0.5
    elif candidate["domain"] and candidate["domain"] in target_domain:
        domain_score = 0.5

    # Tag overlap: Jaccard similarity
    tag_score = 0.0
    if target_tags and candidate["tags"]:
        intersection = set(target_tags) & set(candidate["tags"])
        union = set(target_tags) | set(candidate["tags"])
        tag_score = len(intersection) / len(union) if union else 0.0

    # Case category overlap (new signal)
    case_score = 0.0
    if target_case_categories and candidate.get("cases_handled"):
        intersection = set(target_case_categories) & set(candidate["cases_handled"])
        union = set(target_case_categories) | set(candidate["cases_handled"])
        case_score = len(intersection) / len(union) if union else 0.0

    if target_case_categories:
        return 0.4 * semantic + 0.25 * domain_score + 0.15 * tag_score + 0.2 * case_score
    else:
        return 0.5 * semantic + 0.3 * domain_score + 0.2 * tag_score


def _apply_recency_boost(candidates: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    for c in candidates:
        try:
            ts = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
            days_old = max(0, (now - ts).days)
        except (ValueError, AttributeError):
            days_old = 365
        multiplier = 1.0 / (1.0 + 0.05 * math.log(days_old + 1))
        c["final_score"] = c.get("weighted_score", c["similarity"]) * multiplier
    return candidates


def _mmr_select(
    candidates: list[dict],
    n_results: int = 5,
    lambda_param: float = 0.6,
) -> list[dict]:
    """Maximum Marginal Relevance selection for diversity."""
    if len(candidates) <= n_results:
        return candidates

    selected = []
    remaining = list(candidates)

    remaining.sort(key=lambda c: c.get("final_score", 0), reverse=True)
    selected.append(remaining.pop(0))

    while len(selected) < n_results and remaining:
        best_score = -float("inf")
        best_idx = 0

        for i, candidate in enumerate(remaining):
            relevance = candidate.get("final_score", 0)
            max_redundancy = 0.0
            for sel in selected:
                c_words = set(candidate["prompt"].lower().split()[:50])
                s_words = set(sel["prompt"].lower().split()[:50])
                if c_words and s_words:
                    overlap = len(c_words & s_words) / max(len(c_words | s_words), 1)
                    max_redundancy = max(max_redundancy, overlap)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _reciprocal_rank_fusion(
    chroma_candidates: list[dict],
    bm25_candidates: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    all_candidates: dict[str, dict] = {}

    for rank, c in enumerate(chroma_candidates):
        doc_id = c["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        all_candidates[doc_id] = c

    for rank, c in enumerate(bm25_candidates):
        doc_id = c["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        if doc_id not in all_candidates:
            all_candidates[doc_id] = c

    for doc_id, score in scores.items():
        all_candidates[doc_id]["rrf_score"] = score

    merged = list(all_candidates.values())
    merged.sort(key=lambda c: c.get("rrf_score", 0), reverse=True)
    return merged


# ──────────────── Main Entry Point ────────────────

async def smart_retrieve(
    query: str,
    domain: str,
    state_intent: str = "",
    persona: str = "",
    guardrails: list[str] | None = None,
    escalation_triggers: list[str] | None = None,
    tags: list[str] | None = None,
    case_categories: list[str] | None = None,
    n_results: int = 5,
) -> RetrievalContext:
    """Advanced multi-signal retrieval with BM25 hybrid, MMR, cold start,
    and case-category filtering."""
    guardrails = guardrails or []
    escalation_triggers = escalation_triggers or []
    tags = tags or []
    case_categories = case_categories or []

    try:
        # Step 1: Multi-signal ChromaDB queries (parallel)
        where_strict = _build_where_clause(domain, strict=True)
        where_relaxed = _build_where_clause(domain, strict=False)

        query_a = f"{domain} {persona}".strip()
        query_b = state_intent or query
        query_c = " ".join(guardrails[:3] + escalation_triggers[:3])

        try:
            chroma_results_a = chroma_client.query_collection(
                query_a, n_results=20, where=where_strict
            )
        except Exception:
            chroma_results_a = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        try:
            chroma_results_b = chroma_client.query_collection(
                query_b, n_results=20, where=where_strict
            )
        except Exception:
            chroma_results_b = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        chroma_results_c = None
        if query_c.strip():
            try:
                chroma_results_c = chroma_client.query_collection(
                    query_c, n_results=10, where=where_strict
                )
            except Exception:
                chroma_results_c = None

        # Step 1b: Case-category-specific query (new)
        chroma_results_d = None
        if case_categories:
            case_query = f"{domain} " + " ".join(case_categories)
            try:
                chroma_results_d = chroma_client.query_collection(
                    case_query, n_results=15, where=where_strict
                )
            except Exception:
                chroma_results_d = None

        # Parse into candidate lists
        candidates_a = _parse_chroma_to_candidates(chroma_results_a)
        candidates_b = _parse_chroma_to_candidates(chroma_results_b)
        candidates_c = _parse_chroma_to_candidates(chroma_results_c) if chroma_results_c else []
        candidates_d = _parse_chroma_to_candidates(chroma_results_d) if chroma_results_d else []

        # Merge + deduplicate
        seen_ids = set()
        all_chroma_candidates = []
        for c in candidates_a + candidates_b + candidates_c + candidates_d:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                c["weighted_score"] = _compute_weighted_score(
                    c, domain, tags, case_categories
                )
                all_chroma_candidates.append(c)

        # Subdomain fallback if < 3 results
        if len(all_chroma_candidates) < 3 and domain:
            try:
                fallback_results = chroma_client.query_collection(
                    query_b, n_results=20, where=where_relaxed
                )
            except Exception:
                fallback_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            for c in _parse_chroma_to_candidates(fallback_results):
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    c["weighted_score"] = _compute_weighted_score(
                        c, domain, tags, case_categories
                    )
                    all_chroma_candidates.append(c)

        # Cold Start Detection
        if len(all_chroma_candidates) < 3:
            try:
                unfiltered_results = chroma_client.query_collection(
                    query_b, n_results=20, where=None
                )
            except Exception:
                unfiltered_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            for c in _parse_chroma_to_candidates(unfiltered_results):
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    c["weighted_score"] = _compute_weighted_score(
                        c, domain, tags, case_categories
                    )
                    all_chroma_candidates.append(c)

        is_cold_start = len(all_chroma_candidates) < 3

        if is_cold_start and len(all_chroma_candidates) == 0:
            return RetrievalContext(
                examples=[],
                scores=[],
                retrieval_note="cold start -- no KB reference",
                is_cold_start=True,
            )

        # Step 2: BM25 Hybrid Search
        bm25_candidates = []
        bm25_index, bm25_docs = await _get_or_build_bm25(domain)

        if bm25_index and bm25_docs:
            # Include case categories in BM25 query for better matching
            bm25_query = query_b
            if case_categories:
                bm25_query += " " + " ".join(case_categories)
            tokenized_query = bm25_query.lower().split()
            bm25_scores = bm25_index.get_scores(tokenized_query)

            scored_pairs = list(zip(bm25_docs, bm25_scores))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)

            for doc, score in scored_pairs[:20]:
                if score > 0:
                    max_score = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
                    bm25_candidates.append({
                        "id": doc["id"],
                        "prompt": doc["prompt"],
                        "similarity": 0.0,
                        "domain": domain or "",
                        "state_name": doc.get("state_name", ""),
                        "state_intent": doc.get("state_intent", ""),
                        "tags": doc.get("tags", []),
                        "cases_handled": doc.get("cases_handled", []),
                        "source": "bm25",
                        "timestamp": doc.get("timestamp", ""),
                        "weighted_score": score / max_score if max_score > 0 else 0,
                    })

        # Step 3: RRF Fusion
        if bm25_candidates:
            all_chroma_candidates.sort(
                key=lambda c: c.get("weighted_score", 0), reverse=True
            )
            merged = _reciprocal_rank_fusion(all_chroma_candidates, bm25_candidates)
            for c in merged:
                c["weighted_score"] = c.get("rrf_score", c.get("weighted_score", 0))
        else:
            merged = all_chroma_candidates

        # Step 4: Recency Boost
        merged = _apply_recency_boost(merged)

        # Step 5: MMR Diversity Selection
        selected = _mmr_select(merged, n_results=n_results, lambda_param=0.6)

        # Build output
        examples = [c["prompt"] for c in selected]
        scores = [round(c.get("final_score", c.get("weighted_score", 0)), 3) for c in selected]

        domain_matches = sum(1 for c in selected if c["domain"] == domain)
        if is_cold_start:
            note = "cold start -- limited KB reference, use for structural guidance only"
        elif scores and scores[0] > 0.85:
            note = f"high confidence -- {domain_matches}/{len(selected)} from same domain+persona"
        elif scores and scores[0] > 0.70:
            note = f"moderate confidence -- {domain_matches}/{len(selected)} domain matches"
        else:
            note = "low confidence -- use for structural guidance only"

        return RetrievalContext(
            examples=examples,
            scores=scores,
            retrieval_note=note,
            is_cold_start=is_cold_start,
        )

    except Exception as e:
        logger.error(f"[RetrievalEngine] Error during smart_retrieve: {e}", exc_info=True)
        return RetrievalContext(
            examples=[],
            scores=[],
            retrieval_note="retrieval error -- pattern-only fallback",
            is_cold_start=True,
        )


async def retrieve_for_case_learning(
    domain: str,
    state_name: str,
    state_intent: str,
    case_categories: list[str] | None = None,
    n_results: int = 8,
) -> tuple[list[dict], bool]:
    """Specialized retrieval for KB Case Learner.

    Returns (candidates_with_metadata, is_cold_start) where each candidate
    has the full prompt text plus parsed case metadata.
    """
    case_categories = case_categories or []

    # Standard retrieval
    retrieval_ctx = await smart_retrieve(
        query=f"{domain} {state_name} {state_intent}",
        domain=domain,
        state_intent=state_intent,
        case_categories=case_categories,
        n_results=n_results,
    )

    # Also query structured case data from SQLite
    structured_results = []
    if case_categories:
        structured_results = await sqlite_db.query_kb_by_case_category(
            domain, case_categories, limit=n_results
        )

    # Merge: pair prompt text with structured metadata
    candidates = []
    seen_prompts = set()

    for i, prompt_text in enumerate(retrieval_ctx.examples):
        prompt_key = prompt_text[:200]
        if prompt_key not in seen_prompts:
            seen_prompts.add(prompt_key)
            candidates.append({
                "prompt": prompt_text,
                "score": retrieval_ctx.scores[i] if i < len(retrieval_ctx.scores) else 0,
                "cases_handled": [],
                "case_handling_map": {},
                "variables_used": [],
                "transitions": {},
            })

    for sr in structured_results:
        prompt_key = sr["prompt"][:200]
        if prompt_key not in seen_prompts:
            seen_prompts.add(prompt_key)
            candidates.append({
                "prompt": sr["prompt"],
                "score": 0.5,
                "cases_handled": sr.get("cases_handled", []),
                "case_handling_map": sr.get("case_handling_map", {}),
                "variables_used": sr.get("variables_used", []),
                "transitions": sr.get("transitions", {}),
            })
        else:
            # Enrich existing candidate with structured data
            for c in candidates:
                if c["prompt"][:200] == prompt_key:
                    c["cases_handled"] = sr.get("cases_handled", [])
                    c["case_handling_map"] = sr.get("case_handling_map", {})
                    c["variables_used"] = sr.get("variables_used", [])
                    c["transitions"] = sr.get("transitions", {})
                    break

    return candidates, retrieval_ctx.is_cold_start
