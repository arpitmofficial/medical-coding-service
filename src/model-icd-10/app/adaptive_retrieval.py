"""
Adaptive Retrieval: Auto-detects Database Capabilities
======================================================

This module automatically detects what type of search your Qdrant
database supports and uses the best available method:

1. Full Hybrid Search (dense + sparse vectors with RRF)
2. Named Dense Vectors (dense vectors with named format)
3. Legacy Dense Vectors (single unnamed dense vectors)

Usage:
    from app.adaptive_retrieval import adaptive_retrieve_icd_candidates

    results = await adaptive_retrieve_icd_candidates("patient has fever")
"""

import asyncio
import logging
import time
from typing import Any, Optional

from app.config import FINAL_TOP_N, MIN_SCORE, QDRANT_TOP_K, console_logger
from app.embedding import get_embeddings_batch
from app.execution_analysis import tracker
from app.preprocessing import parse_entities
from app.reranking import rerank_codes
from app.qdrant_rest import (
    search_vectors,
    search_vectors_named_dense_only,
    search_vectors_dense_only,
    QDRANT_URL,
    QDRANT_HEADERS,
    COLLECTION,
)
import httpx

logger = logging.getLogger(__name__)

# Cache the detected capability to avoid repeated detection
_detected_capability: Optional[str] = None


async def detect_database_capability():
    """Detect what search capabilities the database supports."""
    global _detected_capability

    if _detected_capability is not None:
        return _detected_capability

    logger.debug("Auto-detecting database search capabilities...")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Get collection info
            response = await client.get(
                f"{QDRANT_URL}/collections/{COLLECTION}", headers=QDRANT_HEADERS
            )
            response.raise_for_status()

            collection_info = response.json()["result"]
            vectors_config = collection_info["config"]["params"]["vectors"]
            sparse_vectors_config = collection_info["config"]["params"].get(
                "sparse_vectors", {}
            )

            # Check for hybrid search support
            if isinstance(vectors_config, dict) and sparse_vectors_config:
                # Alternative syntax: vectors.dense + sparse_vectors.sparse
                if "dense" in vectors_config and "sparse" in sparse_vectors_config:
                    _detected_capability = "hybrid_alt"
                    console_logger.info(
                        "Hybrid Search ENABLED ( Dense + Sparse vectors with RRF fusion)"
                    )
                else:
                    _detected_capability = "named_dense"
                    console_logger.info(
                        "Dense-only search (consider upgrading to hybrid)"
                    )

            elif isinstance(vectors_config, dict):
                # Check for named vectors format
                if "dense" in vectors_config and "sparse" in vectors_config:
                    _detected_capability = "hybrid"
                    console_logger.info(
                        "Hybrid Search ENABLED ( Dense + Sparse vectors with RRF fusion)"
                    )
                elif "dense" in vectors_config:
                    _detected_capability = "named_dense"
                    console_logger.info(
                        "Dense-only search (consider upgrading to hybrid)"
                    )
                elif "size" in vectors_config:
                    _detected_capability = "legacy_dense"
                    console_logger.info(
                        "Legacy dense search (consider upgrading to hybrid)"
                    )
                else:
                    logger.warning("Unknown vector configuration")
                    _detected_capability = "legacy_dense"  # Fallback
            else:
                _detected_capability = "legacy_dense"
                console_logger.info(
                    "Legacy dense search (consider upgrading to hybrid)"
                )

        except Exception as e:
            logger.error(f"Failed to detect database capability: {e}")
            console_logger.info(
                "Using fallback dense search (database detection failed)"
            )
            _detected_capability = "legacy_dense"

    return _detected_capability


async def adaptive_search_single_entity(
    entity: str,
    capability: str,
    limit: int,
    score_threshold: Optional[float] = None,
    debug: bool = True,
) -> list[dict[str, Any]]:
    """Search for a single entity using the best available method."""

    if capability in ["hybrid", "hybrid_alt"]:
        # Full hybrid search with sparse vectors (both syntaxes use same search format)
        try:
            # Import here to avoid circular imports
            from app.embedding import get_sparse_embeddings_batch
            from app.qdrant_rest import search_vectors_debug

            dense_vectors = await get_embeddings_batch([entity])
            sparse_vectors = await get_sparse_embeddings_batch([entity])

            if debug:
                # Show detailed breakdown of each search stage
                console_logger.info(f"\n Searching for entity: '{entity}'")
                debug_results = await search_vectors_debug(
                    dense_vector=dense_vectors[0],
                    sparse_vector=sparse_vectors[0],
                    limit=limit,
                )
                return debug_results["hybrid"]
            else:
                return await search_vectors(
                    dense_vector=dense_vectors[0],
                    sparse_vector=sparse_vectors[0],
                    limit=limit,
                    score_threshold=score_threshold,
                )
        except Exception as e:
            logger.warning(f"Hybrid search failed for '{entity}': {e}")
            logger.info("Falling back to dense search...")
            capability = "named_dense"  # Fallback

    if capability == "named_dense":
        # Named dense vectors
        try:
            dense_vectors = await get_embeddings_batch([entity])
            return await search_vectors_named_dense_only(
                vector=dense_vectors[0], limit=limit, score_threshold=score_threshold
            )
        except Exception as e:
            logger.warning(f"Named dense search failed for '{entity}': {e}")
            logger.info("Falling back to legacy search...")
            capability = "legacy_dense"  # Fallback

    if capability == "legacy_dense":
        # Legacy single vector search
        dense_vectors = await get_embeddings_batch([entity])
        return await search_vectors_dense_only(
            vector=dense_vectors[0], limit=limit, score_threshold=score_threshold
        )

    raise ValueError(f"Unknown search capability: {capability}")


async def adaptive_retrieve_icd_candidates(
    raw_text: str,
    *,
    qdrant_top_k: int = QDRANT_TOP_K,
    final_top_n: int = FINAL_TOP_N,
    min_score: float = MIN_SCORE,
) -> list[dict[str, Any]]:
    """Adaptive ICD retrieval that auto-detects and uses best search method.

    Args:
        raw_text:      Free-text clinical note or query.
        qdrant_top_k:  Candidates fetched from Qdrant per entity.
        final_top_n:   Number of codes returned to the caller.
        min_score:     Minimum Qdrant similarity score; lower hits are dropped.

    Returns:
        List of up to ``final_top_n`` dicts with code, description, confidence, explanation.
    """
    # ------------------------------------------------------------------
    # Stage 0 — parse clinical entities
    # ------------------------------------------------------------------
    console_logger.info(
        f"Processing: '{raw_text[:50]}{'...' if len(raw_text) > 50 else ''}'"
    )
    logger.debug("Adaptive pipeline start | raw_text length=%d", len(raw_text))
    tracker.reset()
    tracker.pipeline_start = time.perf_counter()
    tracker.start_module("preprocessing.py")
    entities: list[str] = await parse_entities(raw_text)
    tracker.end_module("preprocessing.py")

    if not entities:
        logger.warning("No entities extracted; aborting pipeline.")
        console_logger.error("No medical entities found in text")
        tracker.pipeline_end = time.perf_counter()
        tracker.print_report()
        return []

    # ------------------------------------------------------------------
    # Stage 1 — detect database capabilities
    # ------------------------------------------------------------------
    capability = await detect_database_capability()

    # ------------------------------------------------------------------
    # Stage 2 — adaptive entity search
    # ------------------------------------------------------------------
    logger.debug(f"Searching for {len(entities)} entities using {capability} method...")

    tracker.start_module("embedding.py")
    tracker.start_module("qdrant_rest.py")
    search_tasks = [
        adaptive_search_single_entity(
            entity=entity,
            capability=capability,
            limit=qdrant_top_k,
            score_threshold=None,  # Apply filtering later for consistency
        )
        for entity in entities
    ]

    per_entity_results: list[list[dict[str, Any]]] = await asyncio.gather(*search_tasks)
    tracker.end_module("embedding.py")
    tracker.end_module("qdrant_rest.py")

    # ------------------------------------------------------------------
    # Stage 3 — merge & deduplicate results (same logic as original)
    # ------------------------------------------------------------------
    best_by_code: dict[str, dict[str, Any]] = {}
    for i, hits in enumerate(per_entity_results):
        logger.debug(f"Processing results for entity {i}: {len(hits)} hits")
        for j, hit in enumerate(hits):
            # Debug: Check the actual structure of hit
            logger.debug(f"Hit {j} structure: {type(hit)} = {hit}")

            try:
                # Handle different response formats
                if isinstance(hit, dict):
                    if "payload" in hit and "score" in hit:
                        # Standard format
                        code: str = hit["payload"]["code"]
                        score: float = hit["score"]
                        description: str = hit["payload"]["description"]
                    elif "id" in hit and "payload" in hit and "score" in hit:
                        # Alternative format
                        code: str = hit["payload"]["code"]
                        score: float = hit["score"]
                        description: str = hit["payload"]["description"]
                    else:
                        logger.warning(f"Unexpected hit format: {hit}")
                        continue
                else:
                    logger.error(f"Hit is not a dictionary: {type(hit)} = {hit}")
                    continue

                if code not in best_by_code or score > best_by_code[code]["score"]:
                    best_by_code[code] = {
                        "code": code,
                        "description": description,
                        "score": score,
                    }

            except (KeyError, TypeError) as e:
                logger.error(f"Error processing hit {j}: {e}")
                logger.error(f"Hit data: {hit}")
                continue

    # Filter by minimum similarity score
    candidates = [c for c in best_by_code.values() if c["score"] >= min_score]
    candidates.sort(key=lambda x: x["score"], reverse=True)

    logger.debug(
        f"Search complete | {len(candidates)} candidates above min_score={min_score:.2f} using {capability}"
    )

    if not candidates:
        logger.warning(
            "No candidates above min_score=%.2f; returning empty list.", min_score
        )
        console_logger.error(f"No results above confidence threshold ({min_score:.1%})")
        tracker.pipeline_end = time.perf_counter()
        tracker.print_report()
        return []

    # ------------------------------------------------------------------
    # Stage 4 — LLM re-ranking (returns exactly top 5 with confidence scores)
    # ------------------------------------------------------------------
    # Pass both: extracted entities (what we searched for) and original input (context)
    entities_str = ", ".join(entities)
    logger.debug(
        "LLM re-ranking %d candidates → top %d …", len(candidates), final_top_n
    )
    tracker.start_module("reranking.py")
    reranked = await rerank_codes(
        original_query=entities_str,  # What we searched for
        candidates=candidates,
        original_user_input=raw_text,  # Original user input for context
    )
    tracker.end_module("reranking.py")

    if reranked:
        console_logger.info(
            f" Found {len(reranked)} ICD-10 code(s) | Top result: {reranked[0]['code']} ({reranked[0]['confidence']}%)"
        )
    else:
        console_logger.error("LLM re-ranking returned no results")

    tracker.pipeline_end = time.perf_counter()
    tracker.print_report()

    logger.debug("Adaptive pipeline complete | returning %d results", len(reranked))
    return reranked
