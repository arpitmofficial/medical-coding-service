from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Any
import time

from fastembed import SparseTextEmbedding, TextEmbedding

from app.execution_analysis import tracker

logger = logging.getLogger(__name__)

# --- Dense Embeddings (SapBERT - Local) ---
_sapbert_model = None
_sapbert_tokenizer = None
_device = None
_dense_fallback_model = None


def _get_sapbert_model():
    """Get or initialize the SapBERT model (singleton pattern)."""
    global _sapbert_model, _sapbert_tokenizer, _device
    if _sapbert_model is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ModuleNotFoundError as exc:
            raise ImportError("SapBERT requires torch and transformers") from exc

        logger.debug("Initializing SapBERT model locally...")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _sapbert_tokenizer = AutoTokenizer.from_pretrained(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
        _sapbert_model = AutoModel.from_pretrained(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        ).to(_device)
        _sapbert_model.eval()
        logger.debug("SapBERT model loaded on %s", _device)
    return _sapbert_model, _sapbert_tokenizer, _device


def _get_dense_fallback_model() -> TextEmbedding:
    """Get or initialize a lightweight dense fallback model via FastEmbed."""
    global _dense_fallback_model
    if _dense_fallback_model is None:
        logger.warning(
            "SapBERT dependencies unavailable; falling back to FastEmbed dense model (BAAI/bge-small-en-v1.5)."
        )
        _dense_fallback_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _dense_fallback_model


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


async def get_embeddings_batch(
    texts: list[str], max_retries: int = 3
) -> list[list[float]]:
    """Return a list of dense embedding vectors for the given texts using local SapBERT."""
    if not texts:
        return []

    logger.debug("get_embeddings_batch | embedding %d text(s)", len(texts))

    t0 = time.perf_counter()
    try:
        try:
            import torch

            model, tokenizer, device = _get_sapbert_model()

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            elapsed = time.perf_counter() - t0
            tracker.record_api_call("embedding.py", "SapBERT Local Embeddings", elapsed)
            return embeddings.tolist()
        except (ImportError, ModuleNotFoundError):
            fallback_model = _get_dense_fallback_model()
            dense_embeddings = list(fallback_model.embed(texts))
            result = [emb.tolist() for emb in dense_embeddings]
            elapsed = time.perf_counter() - t0
            tracker.record_api_call("embedding.py", "FastEmbed Dense Fallback", elapsed)
            return result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        tracker.record_api_call(
            "embedding.py", "SapBERT Local Embeddings", elapsed, error=str(e)
        )
        raise


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
            result.append(
                {
                    "indices": sparse_emb.indices.tolist(),
                    "values": sparse_emb.values.tolist(),
                }
            )

        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call("embedding.py", "BM25 Sparse Embeddings", api_elapsed)

        return result
    except Exception as e:
        api_elapsed = time.perf_counter() - t0
        tracker.record_api_call(
            "embedding.py", "BM25 Sparse Embeddings", api_elapsed, error=str(e)
        )
        logger.error("get_sparse_embeddings_batch | error: %s", e)
        raise
