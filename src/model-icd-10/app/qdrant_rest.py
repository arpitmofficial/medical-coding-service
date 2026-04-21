from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

from app.config import QDRANT_URL, QDRANT_HEADERS, QDRANT_TOP_K, console_logger
from app.execution_analysis import tracker

logger = logging.getLogger(__name__)

COLLECTION = "icd10_hybrid"


async def search_vectors(
    dense_vector: list[float],
    sparse_vector: dict[str, Any],
    limit: int = QDRANT_TOP_K,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Query Qdrant using hybrid search (dense + sparse vectors) with RRF fusion.

    Args:
        dense_vector: Dense embedding vector from Jina (768 dimensions).
        sparse_vector: Sparse vector dict with 'indices' and 'values' keys.
        limit: Maximum number of results to return (default: QDRANT_TOP_K).
        score_threshold: Optional minimum similarity score filter applied to 
                        individual searches before fusion.

    Returns:
        List of Qdrant hit dicts with 'id', 'score', and 'payload' keys.
        Scores are fused using Reciprocal Rank Fusion (RRF).
    """
    
    # Build prefetch configurations for both search types
    # NOTE: Qdrant Query API uses 'query' key, NOT 'vector' in prefetch configs
    prefetch_configs = []
    
    # Dense vector search configuration  
    dense_config: Dict[str, Any] = {
        "using": "dense",
        "query": dense_vector,  # Use 'query' not 'vector' for /points/query endpoint
        "limit": limit,
    }
    if score_threshold is not None:
        dense_config["score_threshold"] = score_threshold
    
    prefetch_configs.append(dense_config)
    
    # Sparse vector search configuration
    sparse_config: Dict[str, Any] = {
        "using": "sparse", 
        "query": sparse_vector,  # Use 'query' not 'vector' for /points/query endpoint
        "limit": limit,
    }
    if score_threshold is not None:
        sparse_config["score_threshold"] = score_threshold
        
    prefetch_configs.append(sparse_config)
    
    # Main query payload with RRF fusion
    payload: Dict[str, Any] = {
        "prefetch": prefetch_configs,
        "query": {
            "fusion": "rrf"  # Reciprocal Rank Fusion
        },
        "limit": limit,
        "with_payload": True,
    }

    logger.debug(
        "search_vectors | hybrid search: limit=%d, score_threshold=%s, fusion=rrf", 
        limit, score_threshold
    )

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
            headers=QDRANT_HEADERS,
            json=payload,
        )
        response.raise_for_status()

    # Debug: Log the actual response structure
    response_data = response.json()
    logger.debug(f"Hybrid search response keys: {list(response_data.keys())}")
    logger.debug(f"Full response: {response_data}")
    
    # Handle different response formats for /points/query vs /points/search
    if "result" in response_data:
        if "points" in response_data["result"]:
            # Format: {"result": {"points": [...]}}
            results: list[dict[str, Any]] = response_data["result"]["points"]
        else:
            # Format: {"result": [...]}
            results: list[dict[str, Any]] = response_data["result"]
    else:
        # Direct format: {"points": [...]} or just [...]
        if "points" in response_data:
            results: list[dict[str, Any]] = response_data["points"]
        else:
            results: list[dict[str, Any]] = response_data
    
    logger.debug("search_vectors | hybrid search returned %d results", len(results))
    return results


# Legacy function for backward compatibility (dense-only search)
async def search_vectors_dense_only(
    vector: list[float],
    limit: int = QDRANT_TOP_K,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Legacy dense-only vector search (for backward compatibility).
    
    Note: This function is deprecated. Use search_vectors() with both 
    dense and sparse vectors for better results.
    """
    logger.warning("Using deprecated dense-only search. Consider upgrading to hybrid search.")
    
    payload: dict[str, Any] = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
    }
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold

    logger.debug("search_vectors | limit=%d score_threshold=%s", limit, score_threshold)

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            response = await client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                headers=QDRANT_HEADERS,
                json=payload,
            )
            response.raise_for_status()
    except httpx.TimeoutException:
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("qdrant_rest.py", "Qdrant Search API", api_elapsed,
                                 error="Timeout (> 20 s)")
        logger.error("search_vectors | Qdrant API timeout (> 20 s)")
        raise RuntimeError("Qdrant search API timed out after 20 seconds") from None
    except httpx.ConnectError as e:
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("qdrant_rest.py", "Qdrant Search API", api_elapsed,
                                 error=f"Connection error: {e}")
        logger.error("search_vectors | Qdrant API connection error: %s", e)
        raise RuntimeError(f"Qdrant connection failed: {e}") from e
    except httpx.HTTPStatusError as e:
        api_elapsed = time.perf_counter() - t0
        sc = e.response.status_code
        msg = "Qdrant rate-limited (429)" if sc == 429 else f"Qdrant HTTP {sc} error"
        tracker.record_api_call("qdrant_rest.py", "Qdrant Search API", api_elapsed, error=msg)
        logger.error("search_vectors | %s", msg)
        raise RuntimeError(msg) from e
    api_elapsed = time.perf_counter() - t0
    tracker.record_api_call("qdrant_rest.py", "Qdrant Search API", api_elapsed)

    results: list[dict[str, Any]] = response.json()["result"]
    return results


async def search_vectors_named_dense_only(
    vector: list[float],
    limit: int = QDRANT_TOP_K,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search using named dense vectors only (fallback for partial hybrid setup).
    
    Use this when your collection has named vectors but no sparse vectors yet.
    """
    logger.info("Using named dense vector search")
    
    payload: Dict[str, Any] = {
        "vector": {
            "name": "dense",
            "vector": vector
        },
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

    results: list[dict[str, Any]] = response.json()["result"]
    logger.debug("search_vectors_named_dense | returned %d results", len(results))
    return results


# Search limits for hybrid search
DENSE_SEARCH_LIMIT = 30  # Top 30 from semantic/dense search
SPARSE_SEARCH_LIMIT = 20  # Top 20 from keyword/BM25 search
HYBRID_RESULT_LIMIT = 50  # Results after RRF fusion


async def search_vectors_debug(
    dense_vector: list[float],
    sparse_vector: dict[str, Any],
    limit: int = QDRANT_TOP_K,
) -> dict[str, Any]:
    """Debug function to show results from dense, sparse, and hybrid searches separately.
    
    Uses fixed limits: Dense=30, Sparse=20, Hybrid=50 (no score cutoff).
    Returns a dict with 'dense', 'sparse', and 'hybrid' results for comparison.
    """
    from app.config import console_logger
    
    results = {
        "dense": [],
        "sparse": [],
        "hybrid": []
    }
    
    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Dense-only search (top 30)
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
        
        # 2. Sparse-only search (top 20)
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
        
        # 3. Hybrid search with RRF (combines dense 30 + sparse 20 → top 50)
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
        
        for i, hit in enumerate(hybrid_results[:30], 1):  # Show top 30 of hybrid
            console_logger.info(f"  {i:2d}. [{hit['payload']['code']}] {hit['payload']['description'][:50]} (score: {hit['score']:.4f})")
        
        if len(hybrid_results) > 30:
            console_logger.info(f"  ... and {len(hybrid_results) - 30} more results")
        
        console_logger.info(f"\n{'='*60}")
    
    return results