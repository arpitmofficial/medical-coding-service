"""
Two-stage ICD-10 retrieval pipeline
====================================

Stage 0 — Intent parsing
    LLM extracts a list of specific clinical entities from raw free-text so
    that a vague note like "patient has fever and cold" becomes
    ["acute nasopharyngitis", "fever of unknown origin"], each of which maps
    cleanly to ICD-10 codes.

Stage 1 — Qdrant vector search (per entity)
    Each entity is embedded with jina-embeddings-v2-base-en and the top
    QDRANT_TOP_K candidates are fetched from Qdrant.  Results from all
    entities are merged and deduplicated by ICD-10 code, keeping the highest
    similarity score for each.

Stage 2 — LLM re-ranking
    The merged candidate list is sent back to the LLM for clinical re-ranking.
    The LLM returns the top FINAL_TOP_N codes, each with a confidence (%)
    and a brief explanation.  Candidates below MIN_SCORE are dropped before
    this stage to keep the prompt tight.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config import FINAL_TOP_N, MIN_SCORE, QDRANT_TOP_K
from app.embedding import get_embeddings_batch
from app.execution_analysis import tracker
from app.preprocessing import parse_entities
from app.reranking import rerank_codes
from app.qdrant_rest import search_vectors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def retrieve_icd_candidates(
    raw_text: str,
    *,
    qdrant_top_k: int = QDRANT_TOP_K,
    final_top_n: int = FINAL_TOP_N,
    min_score: float = MIN_SCORE,
) -> list[dict[str, Any]]:
    """Full pipeline: raw clinical text → ranked ICD-10 codes.

    Args:
        raw_text:      Free-text clinical note or query.
        qdrant_top_k:  Candidates fetched from Qdrant per entity.
        final_top_n:   Number of codes returned to the caller.
        min_score:     Minimum Qdrant similarity score; lower hits are dropped.

    Returns:
        List of up to ``final_top_n`` dicts::

            {
                "code":        str,   # e.g. "J06.9"
                "description": str,   # e.g. "Acute upper respiratory infection, unspecified"
                "confidence":  int,   # 0-100 from LLM
                "explanation": str,   # brief clinical rationale
            }
    """
    # ------------------------------------------------------------------
    # Stage 0 — parse clinical entities
    # ------------------------------------------------------------------
    logger.info("Pipeline start | raw_text length=%d", len(raw_text))
    tracker.start_module("preprocessing.py")
    entities: list[str] = await parse_entities(raw_text)
    tracker.end_module("preprocessing.py")

    if not entities:
        logger.warning("No entities extracted; aborting pipeline.")
        return []

    # ------------------------------------------------------------------
    # Stage 1 — embed all entities concurrently, then search Qdrant
    # ------------------------------------------------------------------

    # Embed in a single batched call (one round-trip to Jina)
    logger.info("Embedding %d entities …", len(entities))
    tracker.start_module("embedding.py")
    try:
        vectors: list[list[float]] = await get_embeddings_batch(entities)
    except RuntimeError as exc:
        tracker.end_module("embedding.py")
        logger.error("Embedding stage failed: %s", exc)
        raise
    tracker.end_module("embedding.py")

    # Fan out Qdrant searches concurrently (one coroutine per entity)
    logger.info("Searching Qdrant (top_k=%d) for each entity …", qdrant_top_k)
    tracker.start_module("qdrant_rest.py")
    try:
        qdrant_tasks = [
            search_vectors(vec, limit=qdrant_top_k)
            for vec in vectors
        ]
        per_entity_results: list[list[dict[str, Any]]] = await asyncio.gather(*qdrant_tasks)
    except RuntimeError as exc:
        tracker.end_module("qdrant_rest.py")
        logger.error("Qdrant search stage failed: %s", exc)
        raise
    tracker.end_module("qdrant_rest.py")

    # Merge & deduplicate: keep highest score per ICD code
    best_by_code: dict[str, dict[str, Any]] = {}
    for hits in per_entity_results:
        for hit in hits:
            code: str = hit["payload"]["code"]
            score: float = hit["score"]
            if code not in best_by_code or score > best_by_code[code]["score"]:
                best_by_code[code] = {
                    "code": code,
                    "description": hit["payload"]["description"],
                    "score": score,
                }

    # Filter by minimum similarity score
    candidates = [c for c in best_by_code.values() if c["score"] >= min_score]
    candidates.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        "Qdrant stage complete | %d unique candidates after dedup (min_score=%.2f)",
        len(candidates),
        min_score,
    )

    if not candidates:
        logger.warning("No candidates above min_score=%.2f; returning empty list.", min_score)
        return []

    # ------------------------------------------------------------------
    # Stage 2 — LLM re-ranking
    # ------------------------------------------------------------------
    logger.info("LLM re-ranking %d candidates → top %d …", len(candidates), final_top_n)
    tracker.start_module("reranking.py")
    reranked = await rerank_codes(raw_text, candidates)
    tracker.end_module("reranking.py")

    logger.info("Pipeline complete | returning %d results", len(reranked))
    return reranked
