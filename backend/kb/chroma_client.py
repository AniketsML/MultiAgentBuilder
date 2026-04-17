"""ChromaDB client wrapper with collections for prompts, case-strategies,
domain-filtered retrieval, and Prompt DNA storage."""

import json
import chromadb
from backend.config import CHROMA_PERSIST_DIR
from backend.models.schemas import KBRecord

_client: chromadb.ClientAPI | None = None
_prompts_collection = None
_case_strategies_collection = None
_dna_collection = None


def init_chroma():
    """Initialise the persistent ChromaDB client and collections."""
    global _client, _prompts_collection, _case_strategies_collection, _dna_collection
    _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    _prompts_collection = _client.get_or_create_collection(
        name="prompts_kb",
        metadata={"hnsw:space": "cosine"},
    )
    # Repurposed: stores named case-handling strategies extracted from approved prompts
    _case_strategies_collection = _client.get_or_create_collection(
        name="case_strategies_kb",
        metadata={"hnsw:space": "cosine"},
    )
    # Prompt DNA: stores extracted paradigm principles per KB prompt
    _dna_collection = _client.get_or_create_collection(
        name="dna_kb",
        metadata={"hnsw:space": "cosine"},
    )


def get_prompts_collection():
    return _prompts_collection


def get_case_strategies_collection():
    return _case_strategies_collection


def get_dna_collection():
    return _dna_collection


# ─────────────── Prompt DNA store ───────────────

def upsert_dna(dna_dict: dict):
    """Store a PromptDNA record in the dna_kb collection.

    The document text is a concatenation of all principles for semantic search.
    Metadata carries domain, use_case, confidence scores, and source_prompt_id.
    """
    collection = get_dna_collection()
    source_id = dna_dict.get("source_prompt_id", "")
    domain = dna_dict.get("domain", "")
    use_case = dna_dict.get("use_case", "")

    # Build searchable document: all principles concatenated
    principle_parts = []
    paradigm_names = [
        "structural", "linguistic", "behavioral", "persona",
        "transition", "constraint", "recovery", "rhythm",
    ]
    confidence_map = {}
    for p in paradigm_names:
        dna_field = dna_dict.get(f"{p}_dna", {})
        principles = dna_field.get("principles", [])
        confidence = dna_field.get("confidence", 0.0)
        confidence_map[f"{p}_confidence"] = confidence
        for pr in principles:
            principle_parts.append(f"[{p.upper()}] {pr}")

    document = "\n".join(principle_parts) if principle_parts else "empty DNA"

    metadata = {
        "source_prompt_id": source_id,
        "domain": domain,
        "use_case": use_case,
        "timestamp": dna_dict.get("timestamp", ""),
        **{k: float(v) for k, v in confidence_map.items()},
    }

    doc_id = f"dna_{source_id}"
    collection.upsert(
        ids=[doc_id],
        documents=[document],
        metadatas=[metadata],
    )
    return doc_id


def query_dna(
    query: str,
    domain: str = "",
    n_results: int = 10,
) -> list[dict]:
    """Query DNA store by semantic similarity, optionally filtered by domain.

    Returns list of dicts with id, document, metadata, distance.
    """
    collection = get_dna_collection()
    where = {"domain": domain} if domain else None

    kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception:
        return []

    records = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            records.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
    return records


def get_all_dna_for_domain(domain: str) -> list[dict]:
    """Get all DNA records for a domain. Used by Paradigm Mixer."""
    collection = get_dna_collection()
    try:
        results = collection.get(
            where={"domain": domain},
            include=["documents", "metadatas"],
        )
    except Exception:
        # Fallback: get all if domain filter fails (e.g. empty collection)
        try:
            results = collection.get(include=["documents", "metadatas"])
        except Exception:
            return []

    records = []
    if results["ids"]:
        for i, doc_id in enumerate(results["ids"]):
            records.append({
                "id": doc_id,
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
    return records


def _build_flat_metadata(record: KBRecord) -> dict:
    """Build flat metadata dict for ChromaDB from a KBRecord."""
    metadata = {
        "state_name": record.state_name,
        "context_domain": record.context_domain,
        "state_intent": record.state_intent,
        "tags": ",".join(record.tags),
        "source": record.source,
        "approved_by": record.approved_by,
        "timestamp": record.timestamp,
        "run_id": record.run_id,
        "cases_handled": ",".join(record.cases_handled),
        "variables_used": ",".join(record.variables_used),
    }
    return metadata


def add_to_kb(record: KBRecord):
    """Add a record to the prompts KB collection."""
    collection = get_prompts_collection()
    metadata = _build_flat_metadata(record)
    collection.add(
        ids=[record.id],
        documents=[record.prompt],
        metadatas=[metadata],
    )


def upsert_to_kb(record: KBRecord):
    """Upsert a record into the prompts KB collection."""
    collection = get_prompts_collection()
    metadata = _build_flat_metadata(record)
    collection.upsert(
        ids=[record.id],
        documents=[record.prompt],
        metadatas=[metadata],
    )


def upsert_case_strategy(strategy_id: str, strategy_text: str, domain: str, case_category: str):
    """Store a named case-handling strategy extracted from an approved prompt."""
    collection = get_case_strategies_collection()
    collection.upsert(
        ids=[strategy_id],
        documents=[strategy_text],
        metadatas=[{"domain": domain, "case_category": case_category}],
    )


def query_case_strategies(query: str, domain: str = "", case_category: str = "", n_results: int = 5) -> dict:
    """Query case strategies by semantic similarity with optional domain/category filter."""
    collection = get_case_strategies_collection()
    where = None
    if domain and case_category:
        where = {"$and": [{"domain": domain}, {"case_category": case_category}]}
    elif domain:
        where = {"domain": domain}
    elif case_category:
        where = {"case_category": case_category}

    kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        return collection.query(**kwargs)
    except Exception:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def delete_from_kb(record_id: str):
    """Delete a record from the prompts KB collection."""
    collection = get_prompts_collection()
    collection.delete(ids=[record_id])


def retrieve(
    query: str,
    n_results: int = 5,
    filter_domain: str | None = None,
) -> list[KBRecord]:
    # DEPRECATED: Use retrieval_engine.smart_retrieve() instead.
    collection = get_prompts_collection()

    if filter_domain:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"context_domain": filter_domain},
        )
        if len(results["ids"][0]) < 3:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
            )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )

    records = _parse_chroma_results(results)
    records.sort(key=lambda r: 0 if r.source == "edited" else 1)
    return records


def query_collection(
    query_text: str,
    n_results: int = 20,
    where: dict | None = None,
    include_distances: bool = True,
) -> dict:
    """Low-level ChromaDB query used by retrieval_engine."""
    collection = get_prompts_collection()
    kwargs = {
        "query_texts": [query_text],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def query_by_cases_handled(
    domain: str,
    case_categories: list[str],
    n_results: int = 10,
) -> dict:
    """Query KB for prompts that handle specific case categories."""
    collection = get_prompts_collection()

    # ChromaDB where filter: cases_handled contains the category
    # Since cases_handled is stored as comma-separated string, use $contains
    results_all = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    for category in case_categories:
        try:
            where = {
                "$and": [
                    {"context_domain": {"$eq": domain}},
                    {"cases_handled": {"$contains": category}},
                ]
            }
            result = collection.query(
                query_texts=[f"{domain} {category} handling"],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            # Merge results
            for key in ("ids", "documents", "metadatas", "distances"):
                if result[key] and result[key][0]:
                    results_all[key][0].extend(result[key][0])
        except Exception:
            continue

    return results_all


def _parse_chroma_results(results: dict) -> list[KBRecord]:
    """Parse ChromaDB query results into KBRecord models."""
    records = []
    if not results["ids"] or not results["ids"][0]:
        return records

    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]

        tags = meta.get("tags", "")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        cases_handled = meta.get("cases_handled", "")
        if isinstance(cases_handled, str):
            cases_handled = [c.strip() for c in cases_handled.split(",") if c.strip()]

        variables_used = meta.get("variables_used", "")
        if isinstance(variables_used, str):
            variables_used = [v.strip() for v in variables_used.split(",") if v.strip()]

        records.append(
            KBRecord(
                id=doc_id,
                state_name=meta.get("state_name", ""),
                prompt=doc,
                context_domain=meta.get("context_domain", ""),
                state_intent=meta.get("state_intent", ""),
                tags=tags,
                source=meta.get("source", "generated"),
                approved_by=meta.get("approved_by", ""),
                timestamp=meta.get("timestamp", ""),
                run_id=meta.get("run_id", ""),
                cases_handled=cases_handled,
                variables_used=variables_used,
            )
        )
    return records
