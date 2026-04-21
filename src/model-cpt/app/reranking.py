"""
LLM-Powered Re-ranking for CPT Coding
=====================================

Takes vector search candidates and re-ranks them using clinical reasoning
from an LLM. Returns the most clinically relevant CPT codes with confidence
scores and explanations.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Resolve paths
MODEL_CPT_DIR = Path(__file__).resolve().parent.parent
MODEL_ICD10_DIR = MODEL_CPT_DIR.parent / "model-icd-10"

# Temporarily add ICD-10 to path to load shared modules
_original_path = sys.path.copy()
sys.path.insert(0, str(MODEL_ICD10_DIR))

try:
    _config = importlib.import_module("app.config")
    _execution_analysis = importlib.import_module("app.execution_analysis")
finally:
    # Always restore original import path state
    sys.path = _original_path

LLM_API_KEY = _config.LLM_API_KEY
LLM_MODEL = _config.LLM_MODEL
FINAL_TOP_N = _config.FINAL_TOP_N
console_logger = _config.console_logger
tracker = _execution_analysis.tracker

logger = logging.getLogger(__name__)

# Store original user input for reranking context
_original_user_input: str = ""

# Detect which API to use based on the model name
_is_gemini = LLM_MODEL.startswith("gemini")

if _is_gemini:
    from google import genai
    _gemini_client = genai.Client(api_key=LLM_API_KEY)
else:
    from openai import AsyncOpenAI
    _client = AsyncOpenAI(api_key=LLM_API_KEY)

# ---------------------------------------------------------------------------
# Confidence Score Weights (tune these based on your evaluation data)
# ---------------------------------------------------------------------------
W_RRF = 0.5   # Weight for RRF/vector similarity score
W_LLM = 0.5   # Weight for LLM ranking position


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize_rrf_score(score: float, min_score: float = 0.0, max_score: float = 1.0) -> float:
    """Normalize RRF score to 0-1 range."""
    if max_score == min_score:
        return 0.5
    return max(0.0, min(1.0, (score - min_score) / (max_score - min_score)))


def _llm_rank_to_score(rank: int, total: int) -> float:
    """Convert LLM rank (1-based) to a 0-1 score (higher is better)."""
    if total <= 1:
        return 1.0
    return (total - rank) / (total - 1)


def _calculate_weighted_confidence(
    reranked_results: list[dict[str, Any]],
    original_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Calculate weighted confidence scores combining RRF similarity and LLM ranking.
    
    Formula: confidence = W_RRF * rrf_score + W_LLM * llm_rank_score
    
    Args:
        reranked_results: Results in LLM-ranked order (list of dicts with code, description, explanation)
        original_candidates: Original candidates with RRF scores (list of dicts with code, score)
    
    Returns:
        Results with computed confidence scores (0-100 scale).
    """
    if not reranked_results:
        return []
    
    # Build lookup for original RRF scores
    score_lookup = {c["code"]: c["score"] for c in original_candidates}
    
    # Get min/max scores for normalization
    all_scores = [c["score"] for c in original_candidates if "score" in c]
    min_score = min(all_scores) if all_scores else 0.0
    max_score = max(all_scores) if all_scores else 1.0
    
    total_results = len(reranked_results)
    
    # Calculate weighted confidence for each result
    for i, result in enumerate(reranked_results):
        code = result["code"]
        llm_rank = i + 1  # 1-based rank
        
        # Get original RRF score (default to 0.5 if not found)
        rrf_score = score_lookup.get(code, 0.5)
        normalized_rrf = _normalize_rrf_score(rrf_score, min_score, max_score)
        
        # Convert LLM rank to score
        llm_score = _llm_rank_to_score(llm_rank, total_results)
        
        # Weighted combination (0-1 range)
        raw_confidence = (W_RRF * normalized_rrf) + (W_LLM * llm_score)
        
        # Scale to 0-100
        result["confidence"] = int(raw_confidence * 100)
        
        # Add score info to explanation
        if "explanation" in result and result["explanation"]:
            result["explanation"] = f"{result['explanation']} (Similarity: {rrf_score:.3f}, Rank: #{llm_rank})"
        else:
            result["explanation"] = f"Similarity: {rrf_score:.3f}, Clinical rank: #{llm_rank}"
    
    return reranked_results


def _clean_json_response(content: str) -> str:
    """Remove markdown code fences and extra whitespace from LLM JSON responses."""
    content = content.strip()
    
    # Remove markdown code fences (```json ... ``` or ``` ... ```)
    if content.startswith("```"):
        # Find the end of the opening fence (could be ```json or just ```)
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        
        # Remove closing fence
        if content.endswith("```"):
            content = content[:-3]
    
    return content.strip()


# ---------------------------------------------------------------------------
# Prompt template - MODIFIED FOR CPT CODING
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = (
    "You are a senior medical coder specialising in CPT (Current Procedural Terminology). "
    "Given the ORIGINAL USER INPUT (what the user actually typed), the EXTRACTED PROCEDURAL ENTITY "
    "(what we searched for), and a list of candidate CPT codes, select the TOP 5 most "
    "clinically accurate codes. "
    "Return ONLY a valid JSON array with EXACTLY 5 elements (no more, no less), ordered from most "
    "to least relevant. Each element must have exactly these keys: "
    "\"code\" (string), \"description\" (string), "
    "\"confidence\" (integer 0-100, percent certainty this code applies), "
    "\"explanation\" (one sentence justifying the match).\n\n"
    "Evaluation criteria (in order of priority):\n"
    "1. PROCEDURE TYPE MATCH: Does the CPT code describe the same type of procedure?\n"
    "2. TECHNIQUE: If surgical approach is mentioned (laparoscopic, open, arthroscopic), prioritize matching technique.\n"
    "3. ANATOMICAL SITE: Does the body region/organ match?\n"
    "4. SPECIFICITY: Prefer more specific codes over general ones when the clinical text supports it.\n\n"
    "Confidence scoring guidelines:\n"
    "- 90-100%: Perfect match - procedure type, technique, and anatomy all match\n"
    "- 75-89%: Strong match - procedure type matches with minor differences in specificity\n"
    "- 60-74%: Good match - same general procedure category\n"
    "- 40-59%: Moderate match - related procedure but different specificity or approach\n"
    "- Below 40%: Weak match - only tangentially related\n\n"
    "IMPORTANT: Always return exactly 5 codes. Use the ORIGINAL USER INPUT to understand context "
    "and intent. The extracted entity is what we searched for, but the original input provides "
    "important clinical context for proper code selection."
)


# ---------------------------------------------------------------------------
# Public async function
# ---------------------------------------------------------------------------

async def rerank_codes(
    original_query: str,
    candidates: list[dict[str, Any]],
    original_user_input: str | None = None,
) -> list[dict[str, Any]]:
    """
    Re-rank Qdrant candidates with an LLM and return exactly FINAL_TOP_N (5) codes.

    Args:
        original_query: The extracted procedural entity (what we searched for).
        candidates: List of dicts, each containing at minimum:
                    {"code": str, "description": str, "score": float}
        original_user_input: The original text the user typed (for context).

    Returns:
        List of exactly FINAL_TOP_N (5) dicts:
        [{"code": str, "description": str, "confidence": int, "explanation": str}, ...]
    """
    if not candidates:
        return []

    # Use original_user_input if provided, otherwise fall back to original_query
    user_input = original_user_input if original_user_input else original_query

    logger.debug("rerank_codes | %d candidates to re-rank", len(candidates))
    console_logger.info(f"\n LLM RERANKING (CPT):")
    console_logger.info(f"   Original input: '{user_input}'")
    console_logger.info(f"   Extracted entity: '{original_query}'")
    console_logger.info(f"   Candidates to evaluate: {len(candidates)}")

    # Build a compact representation of candidates for the prompt
    # Only pass code and description - NO scores (let LLM judge purely on clinical merit)
    candidates_text = json.dumps(
        [
            {
                "code": c["code"],
                "description": c["description"],
            }
            for c in candidates
        ],
        indent=2,   
    )

    user_message = (
        f"ORIGINAL USER INPUT (what the user typed):\n\"{user_input}\"\n\n"
        f"EXTRACTED PROCEDURAL ENTITY (what we searched for):\n\"{original_query}\"\n\n"
        f"Candidate CPT codes ({len(candidates)} total):\n{candidates_text}\n\n"
        f"Evaluate each code based purely on clinical accuracy and relevance to the procedure described.\n\n"
        f"Return EXACTLY {FINAL_TOP_N} codes in a JSON array, ordered by relevance."
    )

    input_tokens = output_tokens = total_tokens = None
    error_msg: str | None = None
    content: str | None = None
    t0 = time.perf_counter()

    try:
        if _is_gemini:
            prompt = f"{_RERANK_SYSTEM}\n\n{user_message}"
            response = await asyncio.wait_for(
                _gemini_client.aio.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                    config={"temperature": 0},
                ),
                timeout=60.0,
            )
            content = (response.text or "").strip()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                input_tokens  = getattr(um, "prompt_token_count", None)
                output_tokens = getattr(um, "candidates_token_count", None)
                total_tokens  = getattr(um, "total_token_count", None)
        else:
            response = await asyncio.wait_for(
                _client.chat.completions.create(
                    model=LLM_MODEL,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": _RERANK_SYSTEM},
                        {"role": "user", "content": user_message},
                    ],
                ),
                timeout=60.0,
            )
            content = (response.choices[0].message.content or "").strip()
            if response.usage:
                input_tokens  = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens  = response.usage.total_tokens

    except asyncio.TimeoutError:
        error_msg = "LLM API timeout (> 60 s)"
        logger.error("rerank_codes | %s; falling back to score-based ranking", error_msg)

    except Exception as e:
        err_type = type(e).__name__
        status_code = getattr(e, "status_code", None) or getattr(e, "code", None)
        if status_code == 429 or "RateLimit" in err_type or "ResourceExhausted" in err_type or "quota" in str(e).lower():
            error_msg = f"LLM rate-limited / high traffic ({err_type})"
        else:
            error_msg = f"{err_type}: {e}"
        logger.error("rerank_codes | LLM error — %s; falling back to score-based ranking", error_msg)

    api_elapsed = time.perf_counter() - t0
    tracker.record_api_call(
        "reranking.py", "LLM (rerank_codes)", api_elapsed,
        input_tokens, output_tokens, total_tokens,
        error=error_msg,
    )

    # If the API call failed entirely, fall back to score-based ranking
    if content is None:
        return [
            {
                "code": c["code"],
                "description": c["description"],
                "confidence": round(c["score"] * 100),
                "explanation": "LLM re-ranking unavailable; ranked by vector similarity.",
            }
            for c in sorted(candidates, key=lambda x: x["score"], reverse=True)[:FINAL_TOP_N]
        ]

    
    # Clean markdown fences from response
    content = _clean_json_response(content)

    try:
        reranked: list[dict[str, Any]] = json.loads(content)
        if not isinstance(reranked, list):
            raise ValueError("Expected a JSON array")
        
        # Apply weighted confidence scoring (RRF + LLM rank) with descending order enforcement
        reranked = _calculate_weighted_confidence(reranked[:FINAL_TOP_N], candidates)
        
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "rerank_codes | JSON parse failed (%s); content='%s'; returning top candidates by score",
            exc,
            content[:200]  # Log first 200 chars
        )
        # Graceful degradation: return top FINAL_TOP_N by Qdrant score
        reranked = [
            {
                "code": c["code"],
                "description": c["description"],
                "confidence": round(c["score"] * 100),
                "explanation": "LLM re-ranking unavailable; ranked by vector similarity.",
            }
            for c in sorted(candidates, key=lambda x: x["score"], reverse=True)[:FINAL_TOP_N]
        ]

    return reranked[:FINAL_TOP_N]
