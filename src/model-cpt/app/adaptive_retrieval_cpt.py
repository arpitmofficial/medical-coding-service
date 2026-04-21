"""
Adaptive Retrieval for CPT Codes: Auto-detects Database Capabilities
=====================================================================

This module automatically detects what type of search your Qdrant 
database supports and uses the best available method:

1. Full Hybrid Search (dense + sparse vectors with RRF)
2. Named Dense Vectors (dense vectors with named format)  
3. Legacy Dense Vectors (single unnamed dense vectors)

Usage:
    from adaptive_retrieval_cpt import adaptive_retrieve_cpt_candidates
    
    results = await adaptive_retrieve_cpt_candidates("patient had appendectomy")
"""

import asyncio
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from app.preprocessing import parse_entities as parse_cpt_entities
from app.reranking import rerank_codes as rerank_cpt_codes
from app.config import FINAL_TOP_N, MIN_SCORE, QDRANT_TOP_K, console_logger
from app.embedding import get_embeddings_batch, get_sparse_embeddings_batch
from app.execution_analysis import tracker
from app.qdrant_rest import (
    QDRANT_URL, 
    QDRANT_HEADERS, 
    DENSE_SEARCH_LIMIT, 
    SPARSE_SEARCH_LIMIT, 
    HYBRID_RESULT_LIMIT
)

logger = logging.getLogger(__name__)

# ============================================================================
# CPT COLLECTION NAME (LOCAL OVERRIDE - DO NOT MODIFY qdrant_rest.py)
# ============================================================================
COLLECTION = "cpt_hybrid"

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
                f"{QDRANT_URL}/collections/{COLLECTION}",
                headers=QDRANT_HEADERS
            )
            response.raise_for_status()
            
            collection_info = response.json()["result"]
            vectors_config = collection_info["config"]["params"]["vectors"]
            sparse_vectors_config = collection_info["config"]["params"].get("sparse_vectors", {})
            
            # Check for hybrid search support
            if isinstance(vectors_config, dict) and sparse_vectors_config:
                # Alternative syntax: vectors.dense + sparse_vectors.sparse
                if "dense" in vectors_config and "sparse" in sparse_vectors_config:
                    _detected_capability = "hybrid_alt"
                    console_logger.info("Hybrid Search ENABLED ( Dense + Sparse vectors with RRF fusion)")
                else:
                    _detected_capability = "named_dense" 
                    console_logger.info("Dense-only search (consider upgrading to hybrid)")
                    
            elif isinstance(vectors_config, dict):
                # Check for named vectors format
                if "dense" in vectors_config and "sparse" in vectors_config:
                    _detected_capability = "hybrid"
                    console_logger.info("Hybrid Search ENABLED ( Dense + Sparse vectors with RRF fusion)")
                elif "dense" in vectors_config:
                    _detected_capability = "named_dense"
                    console_logger.info("Dense-only search (consider upgrading to hybrid)")
                elif "size" in vectors_config:
                    _detected_capability = "legacy_dense"
                    console_logger.info("Legacy dense search (consider upgrading to hybrid)")
                else:
                    logger.warning("Unknown vector configuration")
                    _detected_capability = "legacy_dense"  # Fallback
            else:
                _detected_capability = "legacy_dense"
                console_logger.info("Legacy dense search (consider upgrading to hybrid)")
                
        except Exception as e:
            logger.error(f"Failed to detect database capability: {e}")
            console_logger.info("Using fallback dense search (database detection failed)")
            _detected_capability = "legacy_dense"
    
    return _detected_capability


async def _search_vectors_hybrid(
    dense_vector: list[float],
    sparse_vector: dict[str, Any],
    limit: int,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search using dense + sparse vectors with RRF fusion for CPT collection."""
    
    prefetch_configs = [
        {
            "using": "dense",
            "query": dense_vector,
            "limit": limit,
        },
        {
            "using": "sparse", 
            "query": sparse_vector,
            "limit": limit,
        }
    ]
    
    if score_threshold is not None:
        for config in prefetch_configs:
            config["score_threshold"] = score_threshold
    
    payload = {
        "prefetch": prefetch_configs,
        "query": {"fusion": "rrf"},
        "limit": limit,
        "with_payload": True,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
            headers=QDRANT_HEADERS,
            json=payload,
        )
        response.raise_for_status()

    response_data = response.json()
    
    if "result" in response_data:
        if "points" in response_data["result"]:
            results = response_data["result"]["points"]
        else:
            results = response_data["result"]
    else:
        if "points" in response_data:
            results = response_data["points"]
        else:
            results = response_data
    
    return results


async def _search_vectors_named_dense_only(
    vector: list[float],
    limit: int,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search using named dense vectors only for CPT collection."""
    
    payload = {
        "vector": {"name": "dense", "vector": vector},
        "limit": limit,
        "with_payload": True,
    }
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=QDRANT_HEADERS,
            json=payload,
        )
        response.raise_for_status()

    return response.json()["result"]


async def _search_vectors_dense_only(
    vector: list[float],
    limit: int,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Legacy dense-only vector search for CPT collection."""
    
    payload = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
    }
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=QDRANT_HEADERS,
            json=payload,
        )
        response.raise_for_status()

    return response.json()["result"]


async def _search_vectors_debug(
    dense_vector: list[float],
    sparse_vector: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    """Debug function to show results from dense, sparse, and hybrid searches separately."""
    
    results = {
        "dense": [],
        "sparse": [],
        "hybrid": []
    }
    
    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Dense-only search
        console_logger.info(f"\n{'='*60}")
        console_logger.info(f" DENSE (SEMANTIC) SEARCH RESULTS (top {DENSE_SEARCH_LIMIT}):")
        console_logger.info(f"{'='*60}")
        
        dense_payload = {
            "vector": {"name": "dense", "vector": dense_vector},
            "limit": DENSE_SEARCH_LIMIT,
            "with_payload": True
        }
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=QDRANT_HEADERS,
            json=dense_payload
        )
        response.raise_for_status()
        dense_results = response.json()["result"]
        results["dense"] = dense_results
        
        for i, hit in enumerate(dense_results, 1):
            console_logger.info(f"  {i:2d}. [{hit['payload']['code']}] {hit['payload']['description'][:50]} (score: {hit['score']:.4f})")
        
        # 2. Sparse-only search
        console_logger.info(f"\n{'='*60}")
        console_logger.info(f" SPARSE (KEYWORD/BM25) SEARCH RESULTS (top {SPARSE_SEARCH_LIMIT}):")
        console_logger.info(f"{'='*60}")
        
        sparse_payload = {
            "vector": {"name": "sparse", "vector": sparse_vector},
            "limit": SPARSE_SEARCH_LIMIT,
            "with_payload": True
        }
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=QDRANT_HEADERS,
            json=sparse_payload
        )
        response.raise_for_status()
        sparse_results = response.json()["result"]
        results["sparse"] = sparse_results
        
        for i, hit in enumerate(sparse_results, 1):
            console_logger.info(f"  {i:2d}. [{hit['payload']['code']}] {hit['payload']['description'][:50]} (score: {hit['score']:.4f})")
        
        # 3. Hybrid search with RRF
        console_logger.info(f"\n{'='*60}")
        console_logger.info(f" HYBRID (RRF FUSION) SEARCH RESULTS (top {HYBRID_RESULT_LIMIT}):")
        console_logger.info(f"{'='*60}")
        
        hybrid_payload = {
            "prefetch": [
                {"query": dense_vector, "using": "dense", "limit": DENSE_SEARCH_LIMIT},
                {"query": sparse_vector, "using": "sparse", "limit": SPARSE_SEARCH_LIMIT}
            ],
            "query": {"fusion": "rrf"},
            "limit": HYBRID_RESULT_LIMIT,
            "with_payload": True
        }
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
            headers=QDRANT_HEADERS,
            json=hybrid_payload
        )
        response.raise_for_status()
        response_data = response.json()
        
        if "result" in response_data:
            if "points" in response_data["result"]:
                hybrid_results = response_data["result"]["points"]
            else:
                hybrid_results = response_data["result"]
        else:
            hybrid_results = response_data.get("points", response_data)
        
        results["hybrid"] = hybrid_results
        
        for i, hit in enumerate(hybrid_results[:30], 1):
            console_logger.info(f"  {i:2d}. [{hit['payload']['code']}] {hit['payload']['description'][:50]} (score: {hit['score']:.4f})")
        
        if len(hybrid_results) > 30:
            console_logger.info(f"  ... and {len(hybrid_results) - 30} more results")
        
        console_logger.info(f"\n{'='*60}")
    
    return results


async def adaptive_search_single_entity(
    entity: str,
    capability: str,
    limit: int,
    score_threshold: Optional[float] = None,
    debug: bool = True
) -> list[dict[str, Any]]:
    """Search for a single entity using the best available method."""
    
    if capability in ["hybrid", "hybrid_alt"]:
        try:
            dense_vectors = await get_embeddings_batch([entity])
            sparse_vectors = await get_sparse_embeddings_batch([entity])
            
            if debug:
                console_logger.info(f"\n Searching for entity: '{entity}'")
                debug_results = await _search_vectors_debug(
                    dense_vector=dense_vectors[0],
                    sparse_vector=sparse_vectors[0],
                    limit=limit
                )
                return debug_results["hybrid"]
            else:
                return await _search_vectors_hybrid(
                    dense_vector=dense_vectors[0],
                    sparse_vector=sparse_vectors[0],
                    limit=limit,
                    score_threshold=score_threshold
                )
        except Exception as e:
            logger.warning(f"Hybrid search failed for '{entity}': {e}")
            logger.info("Falling back to dense search...")
            capability = "named_dense"
    
    if capability == "named_dense":
        try:
            dense_vectors = await get_embeddings_batch([entity])
            return await _search_vectors_named_dense_only(
                vector=dense_vectors[0],
                limit=limit,
                score_threshold=score_threshold
            )
        except Exception as e:
            logger.warning(f"Named dense search failed for '{entity}': {e}")
            logger.info("Falling back to legacy search...")
            capability = "legacy_dense"
    
    if capability == "legacy_dense":
        dense_vectors = await get_embeddings_batch([entity])
        return await _search_vectors_dense_only(
            vector=dense_vectors[0],
            limit=limit,
            score_threshold=score_threshold
        )
    
    raise ValueError(f"Unknown search capability: {capability}")


async def adaptive_retrieve_cpt_candidates(
    raw_text: str,
    *,
    qdrant_top_k: int = QDRANT_TOP_K,
    final_top_n: int = FINAL_TOP_N,
    min_score: float = MIN_SCORE,
) -> list[dict[str, Any]]:
    """Adaptive CPT retrieval that auto-detects and uses best search method.

    Args:
        raw_text:      Free-text clinical note or query.
        qdrant_top_k:  Candidates fetched from Qdrant per entity.
        final_top_n:   Number of codes returned to the caller.
        min_score:     Minimum Qdrant similarity score; lower hits are dropped.

    Returns:
        List of up to ``final_top_n`` dicts with code, description, confidence, explanation.
    """
    # ------------------------------------------------------------------
    # Stage 0 — parse procedural entities (CPT-specific extraction)
    # ------------------------------------------------------------------
    console_logger.info(f"Processing: '{raw_text[:50]}{'...' if len(raw_text) > 50 else ''}'")
    logger.debug("Adaptive CPT pipeline start | raw_text length=%d", len(raw_text))
    tracker.reset()
    tracker.pipeline_start = time.perf_counter()
    tracker.start_module("preprocessing.py")
    entities: list[str] = await parse_cpt_entities(raw_text) + [raw_text]
    tracker.end_module("preprocessing.py")

    if not entities:
        logger.warning("No entities extracted; aborting pipeline.")
        console_logger.error("No medical procedures found in text")
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
            score_threshold=None
        )
        for entity in entities
    ]
    
    per_entity_results: list[list[dict[str, Any]]] = await asyncio.gather(*search_tasks)
    tracker.end_module("embedding.py")
    tracker.end_module("qdrant_rest.py")

    # ------------------------------------------------------------------
    # Stage 3 — merge & deduplicate results (SAME LOGIC AS ICD)
    # ------------------------------------------------------------------
    best_by_code: dict[str, dict[str, Any]] = {}
    for i, hits in enumerate(per_entity_results):
        logger.debug(f"Processing results for entity {i}: {len(hits)} hits")
        for j, hit in enumerate(hits):
            logger.debug(f"Hit {j} structure: {type(hit)} = {hit}")
            
            try:
                if isinstance(hit, dict):
                    if "payload" in hit and "score" in hit:
                        code: str = hit["payload"]["code"]
                        score: float = hit["score"]
                        description: str = hit["payload"]["description"]
                    elif "id" in hit and "payload" in hit and "score" in hit:
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
        logger.warning("No candidates above min_score=%.2f; returning empty list.", min_score)
        console_logger.error(f"No results above confidence threshold ({min_score:.1%})")
        tracker.pipeline_end = time.perf_counter()
        tracker.print_report()
        return []

    # ------------------------------------------------------------------
    # Stage 4 — LLM re-ranking (CPT-specific reranking)
    # ------------------------------------------------------------------
    entities_str = ", ".join(entities)
    logger.debug("LLM re-ranking %d candidates → top %d …", len(candidates), final_top_n)
    tracker.start_module("reranking.py")
    reranked = await rerank_cpt_codes(
        original_query=entities_str,
        candidates=candidates,
        original_user_input=raw_text
    )
    tracker.end_module("reranking.py")

    if reranked:
        console_logger.info(f" Found {len(reranked)} CPT code(s) | Top result: {reranked[0]['code']} ({reranked[0]['confidence']}%)")
    else:
        console_logger.error("LLM re-ranking returned no results")
        
    tracker.pipeline_end = time.perf_counter()
    tracker.print_report()

    logger.debug("Adaptive CPT pipeline complete | returning %d results", len(reranked))
    return reranked
