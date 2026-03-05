"""
Clinical Text Preprocessing
============================

Extracts structured clinical entities from unstructured free-text notes
using LLM-powered parsing. This solves the "fever/cold" problem where vague
input needs to be mapped to specific medical conditions before ICD-10 lookup.
"""

from __future__ import annotations

import json
import logging

from app.config import LLM_API_KEY, LLM_MODEL

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
