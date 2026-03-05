from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import QDRANT_URL, QDRANT_HEADERS, QDRANT_TOP_K

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

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=QDRANT_HEADERS,
            json=payload,
        )
        response.raise_for_status()

    results: list[dict[str, Any]] = response.json()["result"]
    logger.debug("search_vectors | got %d results", len(results))
    return results