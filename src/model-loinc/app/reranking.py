"""
LLM-Powered Re-ranking for LOINC Coding
=======================================

Re-ranks vector-search candidates for laboratory/observation coding.
Uses LLM plus deterministic facet-aware confidence adjustments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

MODEL_LOINC_DIR = Path(__file__).resolve().parent.parent
MODEL_ICD10_DIR = MODEL_LOINC_DIR.parent / "model-icd-10"

_original_path = sys.path.copy()
sys.path.insert(0, str(MODEL_ICD10_DIR))

try:
    from app.config import FINAL_TOP_N, LLM_API_KEY, LLM_MODEL, console_logger
    from app.execution_analysis import tracker
finally:
    sys.path = _original_path

logger = logging.getLogger(__name__)


# Add Groq support: treat Llama/Mixtral as OpenAI-compatible but with Groq endpoint
_is_gemini = LLM_MODEL.startswith("gemini") or LLM_MODEL.startswith("gemma")
_is_groq = LLM_MODEL.startswith("llama") or LLM_MODEL.startswith("mixtral")

if _is_gemini:
    from google import genai
    # This client works for BOTH Gemini and Gemma models from AI Studio
    _gemini_client = genai.Client(api_key=LLM_API_KEY)
elif _is_groq:
    from openai import AsyncOpenAI
    _client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
else:
    from openai import AsyncOpenAI
    _client = AsyncOpenAI(api_key=LLM_API_KEY)

def _get_openai_client():
    return _client


_RERANK_SYSTEM = (
    "You are a senior medical coder specializing in LOINC. "
    "Given user input, extracted query entities, and candidate LOINC codes, "
    "return exactly 5 best codes in JSON array format. "
    "Each element must include: code, description, confidence, explanation."
)


def _clean_json_response(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1 :]
        if content.endswith("```"):
            content = content[:-3]
    return content.strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _facet_bonus(query_text: str, candidate: dict[str, Any]) -> float:
    """Return facet match bonus in [0, 1] for system/time/method overlap."""
    q = _tokenize(query_text)
    bonus = 0.0

    system = str(candidate.get("system") or "").lower()
    time_aspect = str(candidate.get("time") or "").lower()
    method = str(candidate.get("method") or "").lower()

    if system and any(tok in q for tok in _tokenize(system)):
        bonus += 0.12
    if time_aspect and any(tok in q for tok in _tokenize(time_aspect)):
        bonus += 0.08
    if method and any(tok in q for tok in _tokenize(method)):
        bonus += 0.06
    return min(0.26, bonus)


def _score_lookup(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["code"]: c for c in candidates if "code" in c}


async def rerank_codes(
    original_query: str,
    candidates: list[dict[str, Any]],
    original_user_input: str | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    user_input = original_user_input or original_query
    compact = [
        {
            "code": c.get("code", ""),
            "description": c.get("description", ""),
            "component": c.get("component", ""),
            "system": c.get("system", ""),
            "time": c.get("time", ""),
            "method": c.get("method", ""),
        }
        for c in candidates[:80]
    ]

    user_message = (
        f"ORIGINAL USER INPUT:\n{user_input}\n\n"
        f"EXTRACTED ENTITIES:\n{original_query}\n\n"
        f"CANDIDATES:\n{json.dumps(compact, indent=2)}\n\n"
        f"Return exactly {FINAL_TOP_N} items as JSON array."
    )

    input_tokens = output_tokens = total_tokens = None
    error_msg: str | None = None
    content: str | None = None

    t0 = time.perf_counter()
    try:
        if _is_gemini:
            prompt = f"{_RERANK_SYSTEM}\n\n{user_message}"
            gemini_client = _get_gemini_client()
            response = await asyncio.wait_for(
                gemini_client.aio.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                    config={"temperature": 0},
                ),
                timeout=60.0,
            )
            content = response.text.strip()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                input_tokens = getattr(um, "prompt_token_count", None)
                output_tokens = getattr(um, "candidates_token_count", None)
                total_tokens = getattr(um, "total_token_count", None)
        else:
            openai_client = _get_openai_client()
            response = await asyncio.wait_for(
                openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": _RERANK_SYSTEM},
                        {"role": "user", "content": user_message},
                    ],
                ),
                timeout=60.0,
            )
            content = response.choices[0].message.content.strip()
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
    except asyncio.TimeoutError:
        error_msg = "LLM API timeout (> 60 s)"
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    tracker.record_api_call(
        "reranking.py",
        "LLM (rerank_codes)",
        time.perf_counter() - t0,
        input_tokens,
        output_tokens,
        total_tokens,
        error=error_msg,
    )

    score_map = _score_lookup(candidates)

    if content is None:
        fallback = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)[:FINAL_TOP_N]
        return [
            {
                "code": c["code"],
                "description": c.get("description", ""),
                "confidence": int(max(0, min(100, c.get("score", 0.0) * 100))),
                "explanation": "LLM unavailable; ranked by hybrid similarity.",
            }
            for c in fallback
        ]

    try:
        parsed: list[dict[str, Any]] = json.loads(_clean_json_response(content))
        if not isinstance(parsed, list):
            raise ValueError("Expected list")
    except (json.JSONDecodeError, ValueError):
        parsed = [
            {
                "code": c["code"],
                "description": c.get("description", ""),
                "confidence": int(max(0, min(100, c.get("score", 0.0) * 100))),
                "explanation": "LLM output parse failed; ranked by hybrid similarity.",
            }
            for c in sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)[:FINAL_TOP_N]
        ]

    reranked: list[dict[str, Any]] = []
    for rank, item in enumerate(parsed[:FINAL_TOP_N], start=1):
        code = str(item.get("code", "")).strip()
        if not code or code not in score_map:
            continue
        src = score_map[code]
        base = float(src.get("score", 0.0))
        facet = _facet_bonus(user_input + " " + original_query, src)
        llm_rank_score = (FINAL_TOP_N - rank) / max(1, FINAL_TOP_N - 1)
        confidence = int(max(0, min(100, (0.55 * base + 0.30 * llm_rank_score + 0.15 * facet) * 100)))

        reranked.append(
            {
                "code": code,
                "description": src.get("description", item.get("description", "")),
                "confidence": confidence,
                "explanation": str(item.get("explanation", "")).strip() or "Ranked by semantic and facet match.",
            }
        )

    if not reranked:
        reranked = [
            {
                "code": c["code"],
                "description": c.get("description", ""),
                "confidence": int(max(0, min(100, c.get("score", 0.0) * 100))),
                "explanation": "Fallback ranking by hybrid similarity.",
            }
            for c in sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)[:FINAL_TOP_N]
        ]

    console_logger.info(f"LOINC reranking complete: returned {len(reranked[:FINAL_TOP_N])} codes")
    return reranked[:FINAL_TOP_N]
