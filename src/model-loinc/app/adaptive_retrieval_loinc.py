"""
Adaptive Retrieval for LOINC Codes
==================================

Uses hybrid dense+sparse retrieval with automatic capability detection,
facet-aware candidate scoring, and LLM reranking.
Collection name is fixed to loinc_hybrid.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

MODEL_LOINC_DIR = Path(__file__).resolve().parent.parent
MODEL_ICD10_DIR = MODEL_LOINC_DIR.parent / "model-icd-10"

_original_path = sys.path.copy()
sys.path.insert(0, str(MODEL_ICD10_DIR))

try:
    from app.config import FINAL_TOP_N, MIN_SCORE, QDRANT_TOP_K, QDRANT_HEADERS, QDRANT_URL, console_logger
    from app.embedding import get_embeddings_batch, get_sparse_embeddings_batch
    from app.execution_analysis import tracker
finally:
    sys.path = _original_path

if __package__:
    from .preprocessing import parse_entities
    from .reranking import rerank_codes
else:
    from preprocessing import parse_entities
    from reranking import rerank_codes

logger = logging.getLogger(__name__)

COLLECTION = "loinc_hybrid"
_detected_capability: Optional[str] = None


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _expand_query_entities(entities: list[str], raw_text: str) -> list[str]:
    expansions = {
        "blood sugar": "glucose serum",
        "sugar": "glucose",
        "hba1c": "hemoglobin a1c",
        "cbc": "complete blood count",
        "cmp": "comprehensive metabolic panel",
        "bmp": "basic metabolic panel",
        "lft": "liver function panel",
        "kidney": "renal function panel",
        "cholesterol": "lipid panel",
        "urine": "urinalysis",
    }

    merged = list(entities)
    lower_raw = raw_text.lower()
    for key, value in expansions.items():
        if key in lower_raw:
            merged.append(value)

    merged.append(raw_text)

    seen = set()
    unique: list[str] = []
    for item in merged:
        k = item.lower().strip()
        if k and k not in seen:
            seen.add(k)
            unique.append(item.strip())
    return unique


def _metadata_match_bonus(query_text: str, payload: dict[str, Any]) -> float:
    q = _tokenize(query_text)

    component = str(payload.get("component") or "").lower()
    system = str(payload.get("system") or "").lower()
    prop = str(payload.get("property") or "").lower()
    time_aspect = str(payload.get("time") or "").lower()

    bonus = 0.0
    if component and any(tok in q for tok in _tokenize(component)):
        bonus += 0.10
    if system and any(tok in q for tok in _tokenize(system)):
        bonus += 0.10
    if prop and any(tok in q for tok in _tokenize(prop)):
        bonus += 0.05
    if time_aspect and any(tok in q for tok in _tokenize(time_aspect)):
        bonus += 0.05

    return min(0.30, bonus)


async def detect_database_capability() -> str:
    global _detected_capability
    if _detected_capability is not None:
        return _detected_capability

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(f"{QDRANT_URL}/collections/{COLLECTION}", headers=QDRANT_HEADERS)
            response.raise_for_status()

            params = response.json()["result"]["config"]["params"]
            vectors_config = params.get("vectors", {})
            sparse_vectors_config = params.get("sparse_vectors", {})

            if isinstance(vectors_config, dict) and sparse_vectors_config:
                if "dense" in vectors_config and "sparse" in sparse_vectors_config:
                    _detected_capability = "hybrid_alt"
                else:
                    _detected_capability = "named_dense"
            elif isinstance(vectors_config, dict):
                if "dense" in vectors_config and "sparse" in vectors_config:
                    _detected_capability = "hybrid"
                elif "dense" in vectors_config:
                    _detected_capability = "named_dense"
                else:
                    _detected_capability = "legacy_dense"
            else:
                _detected_capability = "legacy_dense"
        except Exception as exc:
            logger.error("LOINC capability detect failed: %s", exc)
            _detected_capability = "legacy_dense"

    return _detected_capability


async def _search_hybrid(
    dense_vector: list[float],
    sparse_vector: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    payload = {
        "prefetch": [
            {"using": "dense", "query": dense_vector, "limit": limit},
            {"using": "sparse", "query": sparse_vector, "limit": limit},
        ],
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

    data = response.json()
    if "result" in data:
        return data["result"].get("points", data["result"])
    return data.get("points", data)


async def _search_named_dense(vector: list[float], limit: int) -> list[dict[str, Any]]:
    payload = {
        "vector": {"name": "dense", "vector": vector},
        "limit": limit,
        "with_payload": True,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=QDRANT_HEADERS,
            json=payload,
        )
        response.raise_for_status()
    return response.json()["result"]


async def _search_legacy_dense(vector: list[float], limit: int) -> list[dict[str, Any]]:
    payload = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=QDRANT_HEADERS,
            json=payload,
        )
        response.raise_for_status()
    return response.json()["result"]


async def adaptive_search_single_entity(entity: str, capability: str, limit: int) -> list[dict[str, Any]]:
    if capability in ["hybrid", "hybrid_alt"]:
        try:
            dense_vector = (await get_embeddings_batch([entity]))[0]
            sparse_vector = (await get_sparse_embeddings_batch([entity]))[0]
            return await _search_hybrid(dense_vector, sparse_vector, limit)
        except Exception as exc:
            logger.warning("LOINC hybrid search failed for '%s': %s", entity, exc)
            capability = "named_dense"

    if capability == "named_dense":
        try:
            dense_vector = (await get_embeddings_batch([entity]))[0]
            return await _search_named_dense(dense_vector, limit)
        except Exception as exc:
            logger.warning("LOINC named dense search failed for '%s': %s", entity, exc)
            capability = "legacy_dense"

    dense_vector = (await get_embeddings_batch([entity]))[0]
    return await _search_legacy_dense(dense_vector, limit)


async def adaptive_retrieve_loinc_candidates(
    raw_text: str,
    *,
    qdrant_top_k: int = QDRANT_TOP_K,
    final_top_n: int = FINAL_TOP_N,
    min_score: float = MIN_SCORE,
) -> list[dict[str, Any]]:
    """Retrieve and rerank top LOINC candidates."""

    tracker.reset()
    tracker.pipeline_start = time.perf_counter()

    tracker.start_module("preprocessing.py")
    base_entities = await parse_entities(raw_text)
    entities = _expand_query_entities(base_entities, raw_text)
    tracker.end_module("preprocessing.py")

    capability = await detect_database_capability()
    console_logger.info(f"LOINC search capability: {capability}; collection: {COLLECTION}")

    tracker.start_module("embedding.py")
    tracker.start_module("qdrant_rest.py")
    search_tasks = [adaptive_search_single_entity(entity, capability, qdrant_top_k) for entity in entities]
    per_entity_results = await asyncio.gather(*search_tasks)
    tracker.end_module("embedding.py")
    tracker.end_module("qdrant_rest.py")

    best_by_code: dict[str, dict[str, Any]] = {}
    full_query = " ".join(entities)

    for hits in per_entity_results:
        for hit in hits:
            try:
                payload = hit["payload"]
                code = payload["code"]
                base_score = float(hit["score"])
                adjusted_score = base_score + _metadata_match_bonus(full_query, payload)

                candidate = {
                    "code": code,
                    "description": str(payload.get("long_name") or payload.get("description") or payload.get("short_name") or ""),
                    "score": adjusted_score,
                    "base_score": base_score,
                    "component": payload.get("component", ""),
                    "property": payload.get("property", ""),
                    "time": payload.get("time", ""),
                    "system": payload.get("system", ""),
                    "scale": payload.get("scale", ""),
                    "method": payload.get("method", ""),
                    "class": payload.get("class", ""),
                    "short_name": payload.get("short_name", ""),
                    "long_name": payload.get("long_name", ""),
                }

                if code not in best_by_code or adjusted_score > best_by_code[code]["score"]:
                    best_by_code[code] = candidate
            except (KeyError, TypeError, ValueError):
                continue

    candidates = [c for c in best_by_code.values() if c["score"] >= min_score]
    candidates.sort(key=lambda c: c["score"], reverse=True)

    if not candidates:
        tracker.pipeline_end = time.perf_counter()
        tracker.print_report()
        return []

    tracker.start_module("reranking.py")
    reranked = await rerank_codes(
        original_query=", ".join(entities),
        candidates=candidates,
        original_user_input=raw_text,
    )
    tracker.end_module("reranking.py")

    tracker.pipeline_end = time.perf_counter()
    tracker.print_report()

    return reranked[:final_top_n]
