from __future__ import annotations

import logging

import httpx

from app.config import JINA_API_KEY

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

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(_JINA_URL, headers=_JINA_HEADERS, json=payload)
        response.raise_for_status()

    data = response.json()
    return [item["embedding"] for item in data["data"]]