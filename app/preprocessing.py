"""
Clinical Text Preprocessing
============================

Extracts structured clinical entities from unstructured free-text notes
using LLM-powered parsing. This solves the "fever/cold" problem where vague
input needs to be mapped to specific medical conditions before ICD-10 lookup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from app.config import LLM_API_KEY, LLM_MODEL
from app.execution_analysis import tracker

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
# Prompt template
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = (
    "You are a clinical NLP assistant. "
    "Extract every distinct medical condition, diagnosis, or clinically significant "
    "symptom from the text provided by the user. "
    "Return ONLY a valid JSON array of short, precise strings. "
    "Do not include any explanation, markdown fences, or extra keys. "
    "Example output: [\"Type 2 diabetes mellitus\", \"hypertension\", \"chronic kidney disease\"]"
)

# _PARSE_SYSTEM = (
#     "You are a clinical NLP assistant. "
#     "Extract every distinct medical condition, diagnosis, or clinically significant "
#     "Don't change any names to scientific term, keep it as it is."
#     "symptom from the text provided by the user. "
#     "Return ONLY a valid JSON array of short, precise strings. "
#     "Do not include any explanation, markdown fences, or extra keys. "
#     "Example output: [\"Type 2 diabetes mellitus\", \"hypertension\", \"chronic kidney disease\"]"
# )


# ---------------------------------------------------------------------------
# Public async function
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

    input_tokens = output_tokens = total_tokens = None
    error_msg: str | None = None
    content: str | None = None
    t0 = time.perf_counter()

    try:
        if _is_gemini:
            prompt = f"{_PARSE_SYSTEM}\n\nUser input:\n{raw_text}"
            response = await asyncio.wait_for(
                _gemini_client.aio.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                    config={"temperature": 0},
                ),
                timeout=20.0,
            )
            content = response.text.strip()
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
                        {"role": "system", "content": _PARSE_SYSTEM},
                        {"role": "user", "content": raw_text},
                    ],
                ),
                timeout=20.0,
            )
            content = response.choices[0].message.content.strip()
            if response.usage:
                input_tokens  = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens  = response.usage.total_tokens

    except asyncio.TimeoutError:
        error_msg = "LLM API timeout (> 20 s)"
        logger.error("parse_entities | %s; falling back to raw text", error_msg)

    except Exception as e:
        err_type = type(e).__name__
        status_code = getattr(e, "status_code", None) or getattr(e, "code", None)
        if status_code == 429 or "RateLimit" in err_type or "ResourceExhausted" in err_type or "quota" in str(e).lower():
            error_msg = f"LLM rate-limited / high traffic ({err_type})"
        else:
            error_msg = f"{err_type}: {e}"
        logger.error("parse_entities | LLM error — %s; falling back to raw text", error_msg)

    api_elapsed = time.perf_counter() - t0
    tracker.record_api_call(
        "preprocessing.py", "LLM (parse_entities)", api_elapsed,
        input_tokens, output_tokens, total_tokens,
        error=error_msg,
    )

    # If the API call failed entirely, degrade gracefully
    if content is None:
        return [raw_text]

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
