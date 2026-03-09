from __future__ import annotations

import logging
import time

import httpx

from app.config import JINA_API_KEY
from app.execution_analysis import tracker

logger = logging.getLogger(__name__)

_JINA_URL = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v2-base-en"

_JINA_HEADERS = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
}


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts (async).

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of float vectors in the same order as ``texts``.
    """
    if not texts:
        return []

    payload = {
        "input": texts,
        "model": _JINA_MODEL,
    }

    logger.debug("get_embeddings_batch | embedding %d text(s)", len(texts))

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            response = await client.post(_JINA_URL, headers=_JINA_HEADERS, json=payload)
            response.raise_for_status()
    except httpx.TimeoutException:
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("embedding.py", "Jina Embedding API", api_elapsed,
                                 error="Timeout (> 20 s)")
        logger.error("get_embeddings_batch | Jina API timeout (> 20 s)")
        raise RuntimeError("Jina Embedding API timed out after 20 seconds") from None
    except httpx.ConnectError as e:
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("embedding.py", "Jina Embedding API", api_elapsed,
                                 error=f"Connection error: {e}")
        logger.error("get_embeddings_batch | Jina API connection error: %s", e)
        raise RuntimeError(f"Jina Embedding API connection failed: {e}") from e
    except httpx.HTTPStatusError as e:
        api_elapsed = time.perf_counter() - t0
        sc = e.response.status_code
        msg = "Jina API rate-limited (429)" if sc == 429 else f"Jina API HTTP {sc} error"
        tracker.record_api_call("embedding.py", "Jina Embedding API", api_elapsed, error=msg)
        logger.error("get_embeddings_batch | %s", msg)
        raise RuntimeError(msg) from e
    api_elapsed = time.perf_counter() - t0
    tracker.record_api_call("embedding.py", "Jina Embedding API", api_elapsed)

    data = response.json()
    return [item["embedding"] for item in data["data"]]