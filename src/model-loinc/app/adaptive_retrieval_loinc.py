"""
Adaptive Retrieval for LOINC Codes
==================================

Pipeline stages (in order)
--------------------------
1. Entity extraction  — LLM parses the raw clinical text into LOINC facet
                        entities (analyte, specimen, method, …).
2. Hybrid retrieval   — For each entity, dense (SapBERT) + sparse (BM25)
                        vectors are queried against Qdrant (QDRANT_TOP_K per
                        entity, default 50).
3. RRF fusion         — Results from all entity queries are merged via
                        Reciprocal Rank Fusion.
4. Score fusion       — Per-candidate score = α·dense + β·sparse + γ·RRF
                        + metadata_weight·bonus.  Default weights:
                        α=0.55, β=0.35, γ=0.10, metadata_weight=0.50.
5. Pool cap           — Pool is sorted by fused score, then hard-capped to
                        PRE_RERANK_TOP_N (default **50**).  Evaluation shows
                        this retains 82.5 % recall (vs 88.9 % for uncapped)
                        while keeping the re-ranker input tractable.
6. Re-ranking         — (DISABLED) LLM re-ranker will re-order the top-50.
7. Output             — pre-rerank capped pool (default 50) returned to the caller.

Key env-var knobs
-----------------
  QDRANT_TOP_K             candidates fetched per entity (default 50)
  LOINC_PRE_RERANK_TOP_N   pool cap before re-rank / output (default 50)
  FINAL_TOP_N              codes returned to caller (default 5)
  LOINC_FUSION_ALPHA       dense weight  (default 0.55)
  LOINC_FUSION_BETA        sparse weight (default 0.35)
  LOINC_FUSION_GAMMA       RRF weight    (default 0.10)
  LOINC_METADATA_WEIGHT    bonus weight  (default 0.50)
  LOINC_RRF_K              RRF smoothing constant (default 60)
  MIN_SCORE                min fused score to enter pool (default 0.0)
"""

from __future__ import annotations

import asyncio
import logging
import os
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
    from app.config import (
        FINAL_TOP_N,
        MIN_SCORE,
        PRE_RERANK_TOP_N,
        QDRANT_TOP_K,
        QDRANT_HEADERS,
        QDRANT_URL,
        console_logger,
    )
    from app.embedding import get_embeddings_batch, get_sparse_embeddings_batch
    from app.execution_analysis import tracker
finally:
    sys.path = _original_path

if __package__:
    from .preprocessing import parse_entities

    # from .reranking import rerank_codes  # RERANKING DISABLED — re-enable when integrating LLM re-ranker
else:
    from preprocessing import parse_entities

    # from reranking import rerank_codes  # RERANKING DISABLED — re-enable when integrating LLM re-ranker

logger = logging.getLogger(__name__)

COLLECTION = "loinc_hybrid"
_detected_capability: Optional[str] = None

RRF_K = int(os.getenv("LOINC_RRF_K", "60"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DEFAULT_DENSE_THRESHOLD = float(os.getenv("LOINC_DENSE_THRESHOLD", "0.0"))
DEFAULT_SPARSE_THRESHOLD = float(os.getenv("LOINC_SPARSE_THRESHOLD", "0.0"))
DEFAULT_RRF_THRESHOLD = float(os.getenv("LOINC_RRF_THRESHOLD", "0.0"))

DEFAULT_FUSION_ALPHA = float(os.getenv("LOINC_FUSION_ALPHA", "0.55"))
DEFAULT_FUSION_BETA = float(os.getenv("LOINC_FUSION_BETA", "0.35"))
DEFAULT_FUSION_GAMMA = float(os.getenv("LOINC_FUSION_GAMMA", "0.10"))
DEFAULT_METADATA_WEIGHT = float(os.getenv("LOINC_METADATA_WEIGHT", "0.50"))
DEFAULT_USE_DUAL_THRESHOLD = _env_bool("LOINC_USE_DUAL_THRESHOLD", True)

# Cache expanded query entities and merged candidate signals for repeated
# pre-rerank sweeps that only change threshold/fusion parameters.
_PREPROCESSED_QUERY_CACHE: dict[str, list[str]] = {}
_CANDIDATE_SIGNAL_CACHE: dict[
    tuple[str, int], tuple[list[str], dict[str, dict[str, Any]]]
] = {}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize_minmax(value: float | None, min_value: float, max_value: float) -> float:
    if value is None:
        return 0.0
    if max_value <= min_value:
        return 1.0
    return max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))


def _optional_max(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _optional_min(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


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
            response = await client.get(
                f"{QDRANT_URL}/collections/{COLLECTION}", headers=QDRANT_HEADERS
            )
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


async def _search_sparse_only(
    sparse_vector: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    payload = {
        "vector": {"name": "sparse", "vector": sparse_vector},
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


async def adaptive_search_single_entity(
    entity: str, capability: str, limit: int
) -> list[dict[str, Any]]:
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


async def _collect_entity_signals(
    entity: str,
    capability: str,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return dense and sparse hit lists for one entity."""

    dense_vector = (await get_embeddings_batch([entity]))[0]
    sparse_vector: dict[str, Any] | None = None

    dense_hits: list[dict[str, Any]] = []
    sparse_hits: list[dict[str, Any]] = []

    if capability in {"hybrid", "hybrid_alt", "named_dense"}:
        try:
            dense_hits = await _search_named_dense(dense_vector, limit)
        except Exception as exc:
            logger.warning("LOINC named dense search failed for '%s': %s", entity, exc)

    if not dense_hits:
        try:
            dense_hits = await _search_legacy_dense(dense_vector, limit)
        except Exception as exc:
            logger.warning("LOINC legacy dense search failed for '%s': %s", entity, exc)

    if capability in {"hybrid", "hybrid_alt"}:
        try:
            sparse_vector = (await get_sparse_embeddings_batch([entity]))[0]
            sparse_hits = await _search_sparse_only(sparse_vector, limit)
        except Exception as exc:
            logger.warning("LOINC sparse BM25 search failed for '%s': %s", entity, exc)

    return dense_hits, sparse_hits


def _merge_entity_hits(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    full_query: str,
) -> dict[str, dict[str, Any]]:
    """Merge dense/sparse hits for a single entity into per-code metrics."""

    dense_map: dict[str, dict[str, Any]] = {}
    sparse_map: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        payload = hit.get("payload") or {}
        code = payload.get("code")
        if not code:
            continue
        score = float(hit.get("score", 0.0))
        prev = dense_map.get(code)
        if prev is None or score > prev["score"]:
            dense_map[code] = {"hit": hit, "score": score, "rank": rank}

    for rank, hit in enumerate(sparse_hits, start=1):
        payload = hit.get("payload") or {}
        code = payload.get("code")
        if not code:
            continue
        score = float(hit.get("score", 0.0))
        prev = sparse_map.get(code)
        if prev is None or score > prev["score"]:
            sparse_map[code] = {"hit": hit, "score": score, "rank": rank}

    merged: dict[str, dict[str, Any]] = {}
    for code in set(dense_map) | set(sparse_map):
        dense_entry = dense_map.get(code)
        sparse_entry = sparse_map.get(code)

        payload = (
            (dense_entry or {}).get("hit", {}).get("payload")
            or (sparse_entry or {}).get("hit", {}).get("payload")
            or {}
        )

        dense_rank = dense_entry["rank"] if dense_entry else None
        sparse_rank = sparse_entry["rank"] if sparse_entry else None
        dense_score = dense_entry["score"] if dense_entry else None
        sparse_score = sparse_entry["score"] if sparse_entry else None

        rrf_score = 0.0
        if dense_rank is not None:
            rrf_score += 1.0 / (RRF_K + dense_rank)
        if sparse_rank is not None:
            rrf_score += 1.0 / (RRF_K + sparse_rank)

        merged[code] = {
            "code": code,
            "description": str(
                payload.get("long_name")
                or payload.get("description")
                or payload.get("short_name")
                or ""
            ),
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "rrf_score": rrf_score,
            "dense_rank": dense_rank,
            "sparse_rank": sparse_rank,
            "metadata_bonus": _metadata_match_bonus(full_query, payload),
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

    return merged


async def _collect_candidates(
    raw_text: str,
    *,
    qdrant_top_k: int,
    min_score: float,
    track_modules: bool,
    dense_threshold: float | None = None,
    sparse_threshold: float | None = None,
    rrf_threshold: float | None = None,
    use_dual_threshold: bool | None = None,
    fusion_alpha: float | None = None,
    fusion_beta: float | None = None,
    fusion_gamma: float | None = None,
    metadata_weight: float | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Run retrieval up to pre-rerank candidate generation."""

    dense_threshold = (
        DEFAULT_DENSE_THRESHOLD if dense_threshold is None else dense_threshold
    )
    sparse_threshold = (
        DEFAULT_SPARSE_THRESHOLD if sparse_threshold is None else sparse_threshold
    )
    rrf_threshold = DEFAULT_RRF_THRESHOLD if rrf_threshold is None else rrf_threshold
    use_dual_threshold = (
        DEFAULT_USE_DUAL_THRESHOLD if use_dual_threshold is None else use_dual_threshold
    )

    fusion_alpha = DEFAULT_FUSION_ALPHA if fusion_alpha is None else fusion_alpha
    fusion_beta = DEFAULT_FUSION_BETA if fusion_beta is None else fusion_beta
    fusion_gamma = DEFAULT_FUSION_GAMMA if fusion_gamma is None else fusion_gamma
    metadata_weight = (
        DEFAULT_METADATA_WEIGHT if metadata_weight is None else metadata_weight
    )

    fusion_sum = fusion_alpha + fusion_beta + fusion_gamma
    if fusion_sum <= 0:
        fusion_alpha, fusion_beta, fusion_gamma = (
            DEFAULT_FUSION_ALPHA,
            DEFAULT_FUSION_BETA,
            DEFAULT_FUSION_GAMMA,
        )
        fusion_sum = fusion_alpha + fusion_beta + fusion_gamma

    # Normalize dense/sparse/rrf fusion weights to a stable 1.0 sum.
    fusion_alpha /= fusion_sum
    fusion_beta /= fusion_sum
    fusion_gamma /= fusion_sum

    if track_modules:
        tracker.reset()
        tracker.pipeline_start = time.perf_counter()
    cache_key = (raw_text, int(qdrant_top_k))
    entities: list[str]
    best_by_code: dict[str, dict[str, Any]] | None = None

    # Cache is only used for evaluation/sweeps (track_modules=False) to avoid
    # altering runtime tracker timings in the production retrieval path.
    if not track_modules:
        cached = _CANDIDATE_SIGNAL_CACHE.get(cache_key)
        if cached is not None:
            entities, best_by_code = cached

    if best_by_code is None:
        if track_modules:
            tracker.start_module("preprocessing.py")

        if not track_modules and raw_text in _PREPROCESSED_QUERY_CACHE:
            entities = _PREPROCESSED_QUERY_CACHE[raw_text]
        else:
            base_entities = await parse_entities(raw_text)
            entities = _expand_query_entities(base_entities, raw_text)
            if not track_modules:
                _PREPROCESSED_QUERY_CACHE[raw_text] = entities

        if track_modules:
            tracker.end_module("preprocessing.py")

        capability = await detect_database_capability()
        if track_modules:
            console_logger.info(
                f"LOINC search capability: {capability}; collection: {COLLECTION}"
            )
            console_logger.info(
                "LOINC selection config | dense>=%.4f sparse>=%.4f rrf>=%.4f min_score>=%.4f dual=%s "
                "weights(d=%.2f,s=%.2f,r=%.2f,meta=%.2f)",
                dense_threshold,
                sparse_threshold,
                rrf_threshold,
                min_score,
                str(use_dual_threshold),
                fusion_alpha,
                fusion_beta,
                fusion_gamma,
                metadata_weight,
            )
            tracker.start_module("embedding.py")
            tracker.start_module("qdrant_rest.py")

        search_tasks = [
            _collect_entity_signals(entity, capability, qdrant_top_k)
            for entity in entities
        ]
        per_entity_signals = await asyncio.gather(*search_tasks)

        if track_modules:
            tracker.end_module("embedding.py")
            tracker.end_module("qdrant_rest.py")

        best_by_code = {}
        full_query = " ".join(entities)

        for dense_hits, sparse_hits in per_entity_signals:
            per_entity_candidates = _merge_entity_hits(
                dense_hits, sparse_hits, full_query
            )
            for code, candidate in per_entity_candidates.items():
                if code not in best_by_code:
                    best_by_code[code] = candidate
                    continue

                existing = best_by_code[code]
                existing["dense_score"] = _optional_max(
                    existing.get("dense_score"), candidate.get("dense_score")
                )
                existing["sparse_score"] = _optional_max(
                    existing.get("sparse_score"), candidate.get("sparse_score")
                )
                existing["rrf_score"] = max(
                    float(existing.get("rrf_score", 0.0)),
                    float(candidate.get("rrf_score", 0.0)),
                )
                existing["dense_rank"] = _optional_min(
                    existing.get("dense_rank"), candidate.get("dense_rank")
                )
                existing["sparse_rank"] = _optional_min(
                    existing.get("sparse_rank"), candidate.get("sparse_rank")
                )
                existing["metadata_bonus"] = max(
                    float(existing.get("metadata_bonus", 0.0)),
                    float(candidate.get("metadata_bonus", 0.0)),
                )

                if len(str(candidate.get("description", ""))) > len(
                    str(existing.get("description", ""))
                ):
                    existing["description"] = candidate.get("description", "")

                for key in [
                    "component",
                    "property",
                    "time",
                    "system",
                    "scale",
                    "method",
                    "class",
                    "short_name",
                    "long_name",
                ]:
                    if not existing.get(key) and candidate.get(key):
                        existing[key] = candidate[key]

        if not track_modules:
            _CANDIDATE_SIGNAL_CACHE[cache_key] = (entities, best_by_code)

    dense_values = [
        float(c["dense_score"])
        for c in best_by_code.values()
        if c.get("dense_score") is not None
    ]
    sparse_values = [
        float(c["sparse_score"])
        for c in best_by_code.values()
        if c.get("sparse_score") is not None
    ]
    rrf_values = [float(c.get("rrf_score", 0.0)) for c in best_by_code.values()]

    dense_min, dense_max = (
        (min(dense_values), max(dense_values)) if dense_values else (0.0, 0.0)
    )
    sparse_min, sparse_max = (
        (min(sparse_values), max(sparse_values)) if sparse_values else (0.0, 0.0)
    )
    rrf_min, rrf_max = (min(rrf_values), max(rrf_values)) if rrf_values else (0.0, 0.0)

    candidates: list[dict[str, Any]] = []
    for candidate in best_by_code.values():
        dense_score = candidate.get("dense_score")
        sparse_score = candidate.get("sparse_score")
        rrf_score = float(candidate.get("rrf_score", 0.0))

        dense_norm = _normalize_minmax(
            float(dense_score) if dense_score is not None else None,
            dense_min,
            dense_max,
        )
        sparse_norm = _normalize_minmax(
            float(sparse_score) if sparse_score is not None else None,
            sparse_min,
            sparse_max,
        )
        rrf_norm = _normalize_minmax(rrf_score, rrf_min, rrf_max)

        fused_score = (
            (fusion_alpha * dense_norm)
            + (fusion_beta * sparse_norm)
            + (fusion_gamma * rrf_norm)
            + (metadata_weight * float(candidate.get("metadata_bonus", 0.0)))
        )

        candidate["dense_norm"] = dense_norm
        candidate["sparse_norm"] = sparse_norm
        candidate["rrf_norm"] = rrf_norm
        candidate["score"] = fused_score
        candidate["base_score"] = rrf_score

        dense_ok = dense_score is not None and float(dense_score) >= dense_threshold
        sparse_ok = sparse_score is not None and float(sparse_score) >= sparse_threshold
        rrf_ok = rrf_score >= rrf_threshold

        passes_signal_gate = (
            (dense_ok or sparse_ok or rrf_ok) if use_dual_threshold else True
        )
        if passes_signal_gate and fused_score >= min_score:
            candidates.append(candidate)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return entities, candidates


def _apply_pool_cap(
    candidates: list[dict[str, Any]],
    pre_rerank_top_n: int,
) -> list[dict[str, Any]]:
    """
    Hard-cap the fused candidate pool to `pre_rerank_top_n` entries.

    Candidates are already sorted by descending fused_score when passed in.
    This is the validated top-50 strategy: evaluation shows that cutting at
    50 retains 82.5 % recall with zero impact on hit@1 through hit@10,
    while keeping the re-ranker input tractable.
    """
    if len(candidates) > pre_rerank_top_n:
        logger.debug(
            "Pool cap applied: %d → %d candidates (PRE_RERANK_TOP_N=%d)",
            len(candidates),
            pre_rerank_top_n,
            pre_rerank_top_n,
        )
    return candidates[:pre_rerank_top_n]


async def retrieve_loinc_candidates_before_rerank(
    raw_text: str,
    *,
    qdrant_top_k: int = QDRANT_TOP_K,
    pre_rerank_top_n: int = PRE_RERANK_TOP_N,
    min_score: float = MIN_SCORE,
    dense_threshold: float | None = None,
    sparse_threshold: float | None = None,
    rrf_threshold: float | None = None,
    use_dual_threshold: bool | None = None,
    fusion_alpha: float | None = None,
    fusion_beta: float | None = None,
    fusion_gamma: float | None = None,
    metadata_weight: float | None = None,
) -> list[dict[str, Any]]:
    """
    Return the capped candidate pool that would be sent to re-ranking.

    Identical to the retrieval stage of `adaptive_retrieve_loinc_candidates`
    but without the tracker overhead or final formatting — useful for
    evaluation, threshold sweeping, and future re-ranker integration.

    The returned list is sorted by descending fused_score and capped to
    `pre_rerank_top_n` (default 50, env var LOINC_PRE_RERANK_TOP_N).
    """
    _, candidates = await _collect_candidates(
        raw_text,
        qdrant_top_k=qdrant_top_k,
        min_score=min_score,
        track_modules=False,
        dense_threshold=dense_threshold,
        sparse_threshold=sparse_threshold,
        rrf_threshold=rrf_threshold,
        use_dual_threshold=use_dual_threshold,
        fusion_alpha=fusion_alpha,
        fusion_beta=fusion_beta,
        fusion_gamma=fusion_gamma,
        metadata_weight=metadata_weight,
    )
    return _apply_pool_cap(candidates, pre_rerank_top_n)


async def adaptive_retrieve_loinc_candidates(
    raw_text: str,
    *,
    qdrant_top_k: int = QDRANT_TOP_K,
    pre_rerank_top_n: int = PRE_RERANK_TOP_N,
    final_top_n: int = FINAL_TOP_N,
    min_score: float = MIN_SCORE,
    dense_threshold: float | None = None,
    sparse_threshold: float | None = None,
    rrf_threshold: float | None = None,
    use_dual_threshold: bool | None = None,
    fusion_alpha: float | None = None,
    fusion_beta: float | None = None,
    fusion_gamma: float | None = None,
    metadata_weight: float | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve top LOINC candidates using hybrid dense+sparse fusion.

    Pipeline
    --------
    raw_text → entities → hybrid retrieval → RRF fusion → score fusion
             → pool cap (pre_rerank_top_n=50) → [re-rank DISABLED]
             → return capped pool of pre_rerank_top_n codes

    Parameters
    ----------
    raw_text           : free-text clinical order description
    qdrant_top_k       : how many hits Qdrant returns per entity (default 50)
    pre_rerank_top_n   : hard cap on the fused pool BEFORE re-ranking or
                         final output (default 50). Set via env var
                         LOINC_PRE_RERANK_TOP_N.
    final_top_n        : reserved for re-ranker output slicing (default 5);
                         currently unused while reranking is disabled.
    min_score          : minimum fused score to stay in pool (default 0.0)

    Returns
    -------
    List of up to `pre_rerank_top_n` (default 50) candidates, each as
    {code, description, confidence, explanation}, sorted by fused_score
    descending. Re-ranking is disabled and the capped pool is returned as-is.
    """

    entities, candidates = await _collect_candidates(
        raw_text,
        qdrant_top_k=qdrant_top_k,
        min_score=min_score,
        track_modules=True,
        dense_threshold=dense_threshold,
        sparse_threshold=sparse_threshold,
        rrf_threshold=rrf_threshold,
        use_dual_threshold=use_dual_threshold,
        fusion_alpha=fusion_alpha,
        fusion_beta=fusion_beta,
        fusion_gamma=fusion_gamma,
        metadata_weight=metadata_weight,
    )

    if not candidates:
        tracker.pipeline_end = time.perf_counter()
        tracker.print_report()
        return []

    # -------------------------------------------------------------------------
    # RERANKING BLOCK — commented out; restore when integrating LLM re-ranker
    # -------------------------------------------------------------------------
    # tracker.start_module("reranking.py")
    # reranked = await rerank_codes(
    #     original_query=", ".join(entities),
    #     candidates=candidates,
    #     original_user_input=raw_text,
    # )
    # tracker.end_module("reranking.py")
    #
    # tracker.pipeline_end = time.perf_counter()
    # tracker.print_report()
    #
    # return reranked[:final_top_n]
    # -------------------------------------------------------------------------

    # ── Stage 5: pool cap ────────────────────────────────────────────────────
    # Hard-cap to pre_rerank_top_n (default 50) before re-ranking or output.
    # Evaluation result: top-50 retains 82.5 % recall, identical hit@1–hit@10.
    pool = _apply_pool_cap(candidates, pre_rerank_top_n)

    tracker.pipeline_end = time.perf_counter()
    tracker.print_report()

    # ── Stage 6 (DISABLED): LLM re-ranking ──────────────────────────────────
    # RERANKING BLOCK — restore when integrating LLM re-ranker:
    # -------------------------------------------------------------------------
    # tracker.start_module("reranking.py")
    # reranked = await rerank_codes(
    #     original_query=", ".join(entities),
    #     candidates=pool,          # ← pass capped pool, not raw candidates
    #     original_user_input=raw_text,
    # )
    # tracker.end_module("reranking.py")
    # tracker.pipeline_end = time.perf_counter()
    # tracker.print_report()
    # return reranked[:final_top_n]
    # -------------------------------------------------------------------------

    # ── Stage 7: format and return full pool (all 50, score-sorted) ─────────
    # Pipeline ends here.  All `pre_rerank_top_n` candidates are returned in
    # descending fused_score order.  The caller / re-ranker decides how many
    # to consume (e.g. slice [:5] for the final top-5 answer).
    # confidence = fused_score (0–1) scaled to 0–100 integer.
    return [
        {
            "code": c["code"],
            "description": c.get("description", ""),
            "confidence": int(max(0, min(100, c.get("score", 0.0) * 100))),
            "explanation": (
                f"Hybrid fusion (dense+sparse+RRF), pool capped at {pre_rerank_top_n}. "
                f"Dense: {c.get('dense_norm', 0.0):.3f}, "
                f"Sparse: {c.get('sparse_norm', 0.0):.3f}, "
                f"RRF: {c.get('rrf_norm', 0.0):.3f}, "
                f"Metadata bonus: {c.get('metadata_bonus', 0.0):.3f}."
            ),
        }
        for c in pool  # ← all 50, not sliced to final_top_n
    ]
