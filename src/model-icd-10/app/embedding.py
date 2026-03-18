from __future__ import annotations

import asyncio

import logging
from typing import Dict, List, Any
import time

import httpx
from fastembed import SparseTextEmbedding

from app.config import JINA_API_KEY
from app.execution_analysis import tracker

logger = logging.getLogger(__name__)

# --- Dense Embeddings (Jina) ---
_JINA_URL = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v2-base-en"

_JINA_HEADERS = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
}

# --- Sparse Embeddings (BM25-like) ---
# Initialize sparse embedding model (lazy loading)
_sparse_model = None


def _get_sparse_model() -> SparseTextEmbedding:
    """Get or initialize the sparse embedding model (singleton pattern)."""
    global _sparse_model
    if _sparse_model is None:
        logger.debug("Initializing sparse embedding model...")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


async def get_embeddings_batch(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Return a list of dense embedding vectors for the given texts (async) with retries."""
    if not texts:
        return []

    payload = {
        "input": texts,
        "model": _JINA_MODEL,
    }

    logger.debug("get_embeddings_batch | embedding %d text(s)", len(texts))

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(_JINA_URL, headers=_JINA_HEADERS, json=payload)
            response.raise_for_status()
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("embedding.py", "Jina Embeddings API", api_elapsed)
    except Exception as e:
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("embedding.py", "Jina Embeddings API", api_elapsed, error=str(e))
        raise

    data = response.json()
    return [item["embedding"] for item in data["data"]]


async def get_sparse_embeddings_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Return a list of sparse embedding vectors for the given texts.
    
    Uses BM25-style embeddings via FastEmbed for keyword matching.
    Each returned dict has 'indices' and 'values' for sparse vector representation.
    """
    if not texts:
        return []
    
    logger.debug("get_sparse_embeddings_batch | embedding %d text(s)", len(texts))
    
    t0 = time.perf_counter()
    try:
        model = _get_sparse_model()
        
        # Generate sparse embeddings synchronously (FastEmbed is sync)
        sparse_embeddings = list(model.embed(texts))
        
        # Convert to Qdrant-compatible format
        result = []
        for sparse_emb in sparse_embeddings:
            result.append({
                "indices": sparse_emb.indices.tolist(),
                "values": sparse_emb.values.tolist()
            })
        
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("embedding.py", "BM25 Sparse Embeddings", api_elapsed)
        
        return result
    except Exception as e:
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("embedding.py", "BM25 Sparse Embeddings", api_elapsed, error=str(e))
        logger.error("get_sparse_embeddings_batch | error: %s", e)
        raise
    return [item["embedding"] for item in data["data"]]