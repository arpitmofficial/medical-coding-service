from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import QDRANT_URL, QDRANT_HEADERS, QDRANT_TOP_K
from app.execution_analysis import tracker

logger = logging.getLogger(__name__)

COLLECTION = "icd10"


async def search_vectors(
    vector: list[float],
    limit: int = QDRANT_TOP_K,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Query Qdrant for the closest ICD-10 vectors (async).

    Args:
        vector: Embedding vector for the query.
        limit: Maximum number of results to return (default: QDRANT_TOP_K).
        score_threshold: Optional minimum similarity score filter passed to
                         Qdrant so low-quality candidates are dropped early.

    Returns:
        List of Qdrant hit dicts with 'id', 'score', and 'payload' keys.
    """
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
    logger.debug("search_vectors | got %d results", len(results))
    return results