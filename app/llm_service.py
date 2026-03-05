"""
LLM Service — two responsibilities:

1. parse_entities(raw_text)
   Takes a free-text clinical note and returns a deduplicated list of specific
   medical/clinical entities (diagnoses, conditions, symptoms) suitable for
   ICD-10 lookup.

2. rerank_codes(original_query, candidates)
   Takes the original clinical text plus a list of Qdrant candidate dicts
   (each with 'code', 'description', 'score') and returns the top FINAL_TOP_N
   results enriched with a confidence (%) and a brief clinical explanation.

Both functions are async and support both OpenAI and Google Gemini APIs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import LLM_API_KEY, LLM_MODEL, FINAL_TOP_N

logger = logging.getLogger(__name__)

# Detect which API to use based on the model name
_is_gemini = LLM_MODEL.startswith("gemini")

if _is_gemini:
    from google import genai
    _gemini_client = genai.Client(api_key=LLM_API_KEY)
else:
    from openai import AsyncOpenAI
    _client = AsyncOpenAI(api_key=LLM_API_KEY)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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
# Prompt templates
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = (
    "You are a clinical NLP assistant. "
    "Extract every distinct medical condition, diagnosis, or clinically significant "
    "symptom from the text provided by the user. "
    "Return ONLY a valid JSON array of short, precise strings. "
    "Do not include any explanation, markdown fences, or extra keys. "
    "Example output: [\"Type 2 diabetes mellitus\", \"hypertension\", \"chronic kidney disease\"]"
)

_RERANK_SYSTEM = (
    "You are a senior medical coder specialising in ICD-10-CM. "
    "Given the original clinical text and a list of candidate ICD-10 codes with "
    "their descriptions, select the most clinically accurate codes. "
    "Return ONLY a valid JSON array (no markdown, no extra text) ordered from most "
    "to least relevant. Each element must have exactly these keys: "
    "\"code\" (string), \"description\" (string), "
    "\"confidence\" (integer 0-100, percent certainty this code applies), "
    "\"explanation\" (one sentence justifying the match)."
)


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------

async def parse_entities(raw_text: str) -> list[str]:
    """
    Extract a list of specific clinical entities from free-text.

    Args:
        raw_text: Unstructured clinical note or query string.

    Returns:
        Deduplicated list of clinical entity strings, e.g.
        ["Type 2 diabetes mellitus", "hypertension"].
    """
    logger.debug("parse_entities | input length=%d", len(raw_text))

    if _is_gemini:
        prompt = f"{_PARSE_SYSTEM}\n\nUser input:\n{raw_text}"
        response = await _gemini_client.aio.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config={"temperature": 0},
        )
        content = response.text.strip()
    else:
        response = await _client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _PARSE_SYSTEM},
                {"role": "user", "content": raw_text},
            ],
        )
        content = response.choices[0].message.content.strip()
    
    logger.debug("parse_entities | raw LLM response: %s", content)
    
    # Clean markdown fences from response
    content = _clean_json_response(content)

    try:
        entities: list[str] = json.loads(content)
        if not isinstance(entities, list):
            raise ValueError("Expected a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "parse_entities | JSON parse failed (%s); content='%s'; falling back to raw text",
            exc,
            content[:200]  # Log first 200 chars
        )
        # Graceful degradation: treat the whole text as one entity
        entities = [raw_text]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        key = e.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(e.strip())

    logger.info("parse_entities | extracted %d entities: %s", len(unique), unique)
    return unique


async def rerank_codes(
    original_query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Re-rank Qdrant candidates with an LLM and return the top FINAL_TOP_N.

    Args:
        original_query: The original clinical text (used for clinical context).
        candidates: List of dicts, each containing at minimum:
                    {"code": str, "description": str, "score": float}

    Returns:
        List of up to FINAL_TOP_N dicts:
        [{"code": str, "description": str, "confidence": int, "explanation": str}, ...]
    """
    if not candidates:
        return []

    logger.debug("rerank_codes | %d candidates to re-rank", len(candidates))

    # Build a compact representation of candidates for the prompt
    candidates_text = json.dumps(
        [{"code": c["code"], "description": c["description"]} for c in candidates],
        indent=2,
    )

    user_message = (
        f"Clinical text:\n{original_query}\n\n"
        f"Candidate ICD-10 codes ({len(candidates)} total):\n{candidates_text}\n\n"
        f"Return the top {FINAL_TOP_N} most appropriate codes."
    )

    if _is_gemini:
        prompt = f"{_RERANK_SYSTEM}\n\n{user_message}"
        response = await _gemini_client.aio.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config={"temperature": 0},
        )
        content = response.text.strip()
    else:
        response = await _client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content.strip()

    logger.debug("rerank_codes | raw LLM response: %s", content)
    
    # Clean markdown fences from response
    content = _clean_json_response(content)

    try:
        reranked: list[dict[str, Any]] = json.loads(content)
        if not isinstance(reranked, list):
            raise ValueError("Expected a JSON array")
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
