"""ChromaDB client wrapper with two collections and domain-filtered retrieval."""

import chromadb
from backend.config import CHROMA_PERSIST_DIR
from backend.models.schemas import KBRecord

_client: chromadb.ClientAPI | None = None
_prompts_collection = None
_patterns_collection = None


def init_chroma():
    """Initialise the persistent ChromaDB client and collections."""
    global _client, _prompts_collection, _patterns_collection
    _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    _prompts_collection = _client.get_or_create_collection(
        name="prompts_kb",
        metadata={"hnsw:space": "cosine"},
    )
    _patterns_collection = _client.get_or_create_collection(
        name="patterns_kb",
        metadata={"hnsw:space": "cosine"},
    )


def get_prompts_collection():
    return _prompts_collection


def get_patterns_collection():
    return _patterns_collection



def _build_flat_metadata(record: KBRecord) -> dict:
    """Build flat metadata dict for ChromaDB from a KBRecord."""
    metadata = record.model_dump(exclude={"prompt", "id"})
    metadata["tags"] = ",".join(metadata["tags"])
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
    """Upsert a record into the prompts KB collection.
    Used on approval so re-approvals don't create duplicates and
    the semantic index stays fresh for subsequent runs."""
    collection = get_prompts_collection()
    metadata = _build_flat_metadata(record)
    collection.upsert(
        ids=[record.id],
        documents=[record.prompt],
        metadatas=[metadata],
    )


def add_pattern(pattern_id: str, pattern_text: str, domain: str):
    """Add a style pattern to the patterns KB collection."""
    collection = get_patterns_collection()
    collection.add(
        ids=[pattern_id],
        documents=[pattern_text],
        metadatas=[{"domain": domain}],
    )


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
    # Kept for backward compatibility only.
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
    """Low-level ChromaDB query used by retrieval_engine.
    Returns raw ChromaDB response with distances."""
    collection = get_prompts_collection()
    kwargs = {
        "query_texts": [query_text],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


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
            )
        )
    return records
