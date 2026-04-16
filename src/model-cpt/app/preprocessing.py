"""
Clinical Text Preprocessing for CPT Coding
===========================================

Extracts structured procedural entities from unstructured free-text notes
using LLM-powered parsing. Optimized for CPT (procedural) coding extraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Resolve paths
MODEL_CPT_DIR = Path(__file__).resolve().parent.parent
MODEL_ICD10_DIR = MODEL_CPT_DIR.parent / "model-icd-10"

# Temporarily add ICD-10 to path to load shared modules
_original_path = sys.path.copy()
sys.path.insert(0, str(MODEL_ICD10_DIR))

from app.config import LLM_API_KEY, LLM_MODEL, console_logger
from app.execution_analysis import tracker

# Restore original path
sys.path = _original_path

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
# Prompt template - MODIFIED FOR CPT PROCEDURE EXTRACTION
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = (
    "You are a clinical NLP assistant specializing in CPT (Current Procedural Terminology) coding preparation. "
    "Extract every distinct medical PROCEDURE, surgery, treatment, diagnostic test, or service from the text. "
    "IMPORTANT RULES:\n"
    "1. Retain Anatomical Context: NEVER isolate a procedure from its anatomical site or laterality (left/right). "
    "Extract 'appendectomy' not just 'removal', 'CT scan of abdomen' not just 'CT scan'.\n"
    "2. Keep Procedural Specificity: Preserve technique details when mentioned "
    "(e.g., 'laparoscopic cholecystectomy', 'open reduction internal fixation').\n"
    "3. Capture Service Details: Include relevant modifiers like 'initial', 'follow-up', 'consultation'.\n"
    "4. Convert colloquial/layman terms to standard medical procedure terminology:\n"
    "   - 'appendix removal' → 'appendectomy'\n"
    "   - 'gallbladder surgery' → 'cholecystectomy'\n"
    "   - 'knee replacement' → 'total knee arthroplasty'\n"
    "   - 'heart bypass' → 'coronary artery bypass graft'\n"
    "   - 'CAT scan' or 'CT' → 'CT scan' (with body part if mentioned)\n"
    "   - 'MRI' → 'MRI' (with body part if mentioned)\n"
    "   - 'blood test' → specify test type if inferable, else 'laboratory blood analysis'\n"
    "   - 'X-ray' → 'radiograph' (with body part if mentioned)\n"
    "5. Do NOT extract diagnoses or conditions - ONLY procedures, tests, and services.\n"
    "Return ONLY a valid JSON array of descriptive procedural strings. Do not include markdown fences or extra text.\n"
    "Example output: [\"appendectomy\", \"CT scan of abdomen\", \"laparoscopic cholecystectomy\"]"
)


# ---------------------------------------------------------------------------
# Public async function
# ---------------------------------------------------------------------------

async def parse_entities(raw_text: str) -> list[str]:
    """
    Extract a list of specific procedural entities from free-text.

    Args:
        raw_text: Unstructured clinical note or query string.

    Returns:
        Deduplicated list of procedural entity strings, e.g.
        ["appendectomy", "CT scan of abdomen"].
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

    logger.debug("parse_entities | extracted %d entities: %s", len(unique), unique)
    
    # Show extracted entities in console
    console_logger.info(f"🧠 LLM Entity Extraction (CPT): {unique}")
    
    return unique
