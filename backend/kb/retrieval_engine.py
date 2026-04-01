"""Advanced Retrieval Engine — Multi-signal + BM25 Hybrid + MMR + Cold Start.

Replaces the naive single-vector query in chroma_client.retrieve().
All agent retrieval goes through smart_retrieve().
"""

import math
import asyncio
import logging
from datetime import datetime, timezone

from rank_bm25 import BM25Okapi

from backend.kb import chroma_client
from backend.kb import sqlite_db
from backend.models.schemas import RetrievalContext

logger = logging.getLogger(__name__)

# ──────────────── BM25 Index Cache ────────────────
# Built once per domain on first query, invalidated on upsert.
_bm25_cache: dict[str, dict] = {}
# Format: { domain: { "index": BM25Okapi, "docs": [...], "timestamp": "..." } }


def invalidate_bm25_cache(domain: str):
    """Called after upsert_to_kb() to force a rebuild on next query."""
    _bm25_cache.pop(domain, None)
    _bm25_cache.pop("__all__", None)  # Also invalidate unfiltered cache


async def _get_or_build_bm25(domain: str | None) -> tuple:
    """Get cached BM25 index or build a new one.
    Returns (bm25_index, doc_list) or (None, []) if no records."""
    cache_key = domain or "__all__"

    # Check if cache is still valid
    if cache_key in _bm25_cache:
        cached = _bm25_cache[cache_key]
        if domain:
            latest_ts = await sqlite_db.get_last_upsert_timestamp(domain)
            if latest_ts == cached.get("timestamp"):
                return cached["index"], cached["docs"]
        else:
            return cached["index"], cached["docs"]

    # Build fresh index
    records = await sqlite_db.get_all_prompts_for_bm25(domain)
    if not records:
        return None, []

    # Tokenize: combine prompt + state_intent for each record
    corpus = []
    for r in records:
        text = f"{r['prompt']} {r['state_intent']}"
        corpus.append(text.lower().split())

    bm25 = BM25Okapi(corpus)

    # Cache it
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
    """Build ChromaDB where clause with metadata pre-filter."""
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
        # Relaxed: match domain substring (for subdomain fallback)
        return {"context_domain": {"$eq": domain}}


def _parse_chroma_to_candidates(results: dict) -> list[dict]:
    """Parse raw ChromaDB response into a list of candidate dicts."""
    candidates = []
    if not results["ids"] or not results["ids"][0]:
        return candidates

    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        distance = results["distances"][0][i] if results.get("distances") else 1.0

        # ChromaDB cosine distance → similarity (1 - distance)
        similarity = max(0.0, 1.0 - distance)

        tags = meta.get("tags", "")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        candidates.append({
            "id": doc_id,
            "prompt": doc,
            "similarity": similarity,
            "domain": meta.get("context_domain", ""),
            "state_name": meta.get("state_name", ""),
            "state_intent": meta.get("state_intent", ""),
            "tags": tags,
            "source": meta.get("source", "generated"),
            "timestamp": meta.get("timestamp", ""),
        })

    return candidates


def _compute_weighted_score(
    candidate: dict,
    target_domain: str,
    target_tags: list[str],
) -> float:
    """Compute weighted score: 0.5×semantic + 0.3×domain_match + 0.2×tag_overlap."""
    semantic = candidate["similarity"]

    # Domain match: 1.0 if exact, 0.5 if partial substring, 0.0 if none
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

    return 0.5 * semantic + 0.3 * domain_score + 0.2 * tag_score


def _apply_recency_boost(candidates: list[dict]) -> list[dict]:
    """Apply recency multiplier: newer records score higher.
    Formula: recency_multiplier = 1 / (1 + 0.05 × log(days_since_added + 1))
    """
    now = datetime.now(timezone.utc)

    for c in candidates:
        try:
            ts = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
            days_old = max(0, (now - ts).days)
        except (ValueError, AttributeError):
            days_old = 365  # Default to old if timestamp is bad

        multiplier = 1.0 / (1.0 + 0.05 * math.log(days_old + 1))
        c["final_score"] = c.get("weighted_score", c["similarity"]) * multiplier

    return candidates


def _mmr_select(
    candidates: list[dict],
    n_results: int = 5,
    lambda_param: float = 0.6,
) -> list[dict]:
    """Maximum Marginal Relevance selection for diversity.
    From top-N candidates, iteratively selects results that maximize:
    λ × sim(candidate, query) - (1-λ) × max_sim(candidate, already_selected)
    """
    if len(candidates) <= n_results:
        return candidates

    selected = []
    remaining = list(candidates)

    # Select the highest-scored candidate first
    remaining.sort(key=lambda c: c.get("final_score", 0), reverse=True)
    selected.append(remaining.pop(0))

    while len(selected) < n_results and remaining:
        best_score = -float("inf")
        best_idx = 0

        for i, candidate in enumerate(remaining):
            relevance = candidate.get("final_score", 0)

            # Compute max similarity to already-selected (using prompt text overlap as proxy)
            max_redundancy = 0.0
            for sel in selected:
                # Simple word overlap as diversity metric
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
    """Fuse ChromaDB (semantic) and BM25 (keyword) results with RRF.
    score = 1/(k + rank_chroma) + 1/(k + rank_bm25)
    """
    scores: dict[str, float] = {}
    all_candidates: dict[str, dict] = {}

    # Score from chroma ranking
    for rank, c in enumerate(chroma_candidates):
        doc_id = c["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        all_candidates[doc_id] = c

    # Score from BM25 ranking
    for rank, c in enumerate(bm25_candidates):
        doc_id = c["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        if doc_id not in all_candidates:
            all_candidates[doc_id] = c

    # Apply RRF scores
    for doc_id, score in scores.items():
        all_candidates[doc_id]["rrf_score"] = score

    # Sort by RRF score
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
    n_results: int = 5,
) -> RetrievalContext:
    """Advanced multi-signal retrieval with BM25 hybrid, MMR, and cold start.

    Pipeline never crashes on retrieval failure — returns cold-start fallback.
    """
    guardrails = guardrails or []
    escalation_triggers = escalation_triggers or []
    tags = tags or []

    try:
        # ── Step 1: Multi-signal ChromaDB queries (parallel) ──
        where_strict = _build_where_clause(domain, strict=True)
        where_relaxed = _build_where_clause(domain, strict=False)

        # Build 3 query strings
        query_a = f"{domain} {persona}".strip()                         # stylistic
        query_b = state_intent or query                                  # functional
        query_c = " ".join(guardrails[:3] + escalation_triggers[:3])    # constraints

        # Fire all 3 queries — wrapped in try/except for empty-collection safety
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

        # Parse into candidate lists
        candidates_a = _parse_chroma_to_candidates(chroma_results_a)
        candidates_b = _parse_chroma_to_candidates(chroma_results_b)
        candidates_c = _parse_chroma_to_candidates(chroma_results_c) if chroma_results_c else []

        # Merge + deduplicate
        seen_ids = set()
        all_chroma_candidates = []
        for c in candidates_a + candidates_b + candidates_c:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                c["weighted_score"] = _compute_weighted_score(c, domain, tags)
                all_chroma_candidates.append(c)

        # ── Step 1b: Subdomain fallback if < 3 results ──
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
                    c["weighted_score"] = _compute_weighted_score(c, domain, tags)
                    all_chroma_candidates.append(c)

        # ── Cold Start Detection ──
        if len(all_chroma_candidates) < 3:
            # Relax to full KB search matching only on state_intent
            try:
                unfiltered_results = chroma_client.query_collection(
                    query_b, n_results=20, where=None
                )
            except Exception:
                unfiltered_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            for c in _parse_chroma_to_candidates(unfiltered_results):
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    c["weighted_score"] = _compute_weighted_score(c, domain, tags)
                    all_chroma_candidates.append(c)

        is_cold_start = len(all_chroma_candidates) < 3

        if is_cold_start and len(all_chroma_candidates) == 0:
            return RetrievalContext(
                examples=[],
                scores=[],
                retrieval_note="cold start — no KB reference",
                is_cold_start=True,
            )

        # ── Step 2: BM25 Hybrid Search ──
        bm25_candidates = []
        bm25_index, bm25_docs = await _get_or_build_bm25(domain)

        if bm25_index and bm25_docs:
            tokenized_query = query_b.lower().split()
            bm25_scores = bm25_index.get_scores(tokenized_query)

            # Get top-20 by BM25
            scored_pairs = list(zip(bm25_docs, bm25_scores))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)

            for doc, score in scored_pairs[:20]:
                if score > 0:
                    max_score = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
                    bm25_candidates.append({
                        "id": doc["id"],
                        "prompt": doc["prompt"],
                        "similarity": 0.0,  # No vector sim for BM25 results
                        "domain": domain or "",
                        "state_name": doc.get("state_name", ""),
                        "state_intent": doc.get("state_intent", ""),
                        "tags": doc.get("tags", []),
                        "source": "bm25",
                        "timestamp": doc.get("timestamp", ""),
                        "weighted_score": score / max_score if max_score > 0 else 0,
                    })

        # ── Step 3: RRF Fusion ──
        if bm25_candidates:
            # Sort chroma candidates by weighted_score for RRF ranking
            all_chroma_candidates.sort(
                key=lambda c: c.get("weighted_score", 0), reverse=True
            )
            merged = _reciprocal_rank_fusion(all_chroma_candidates, bm25_candidates)
            # Use RRF score as the unified score
            for c in merged:
                c["weighted_score"] = c.get("rrf_score", c.get("weighted_score", 0))
        else:
            merged = all_chroma_candidates

        # ── Step 4: Recency Boost ──
        merged = _apply_recency_boost(merged)

        # ── Step 5: MMR Diversity Selection ──
        selected = _mmr_select(merged, n_results=n_results, lambda_param=0.6)

        # ── Build output ──
        examples = [c["prompt"] for c in selected]
        scores = [round(c.get("final_score", c.get("weighted_score", 0)), 3) for c in selected]

        # Generate retrieval note
        domain_matches = sum(1 for c in selected if c["domain"] == domain)
        if is_cold_start:
            note = "cold start — limited KB reference, use for structural guidance only"
        elif scores and scores[0] > 0.85:
            note = f"high confidence — {domain_matches}/{len(selected)} from same domain+persona"
        elif scores and scores[0] > 0.70:
            note = f"moderate confidence — {domain_matches}/{len(selected)} domain matches"
        else:
            note = f"low confidence — use for structural guidance only"

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
            retrieval_note="retrieval error — pattern-only fallback",
            is_cold_start=True,
        )
