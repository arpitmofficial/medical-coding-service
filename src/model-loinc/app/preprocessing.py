"""
Clinical Text Preprocessing for LOINC Coding
============================================

Extracts structured lab and observation entities from free-text notes.
Includes lightweight heuristic normalization plus LLM extraction for robustness.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

MODEL_LOINC_DIR = Path(__file__).resolve().parent.parent
MODEL_ICD10_DIR = MODEL_LOINC_DIR.parent / "model-icd-10"

_original_path = sys.path.copy()
sys.path.insert(0, str(MODEL_ICD10_DIR))

try:
    from app.config import LLM_API_KEY, LLM_MODEL, console_logger
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
        api_key=LLM_API_KEY, base_url="https://api.groq.com/openai/v1"
    )
else:
    from openai import AsyncOpenAI

    _client = AsyncOpenAI(api_key=LLM_API_KEY)


def _get_openai_client():
    return _client


def _get_gemini_client():
    return _gemini_client


_LAY_TERM_MAP = {
    "blood sugar": "glucose",
    "sugar test": "glucose",
    "hba1c": "hemoglobin a1c",
    "cbc": "complete blood count",
    "cmp": "comprehensive metabolic panel",
    "bmp": "basic metabolic panel",
    "lft": "liver function panel",
    "kidney function": "renal function panel",
    "cholesterol": "lipid panel",
    "thyroid": "thyroid stimulating hormone",
    "urine test": "urinalysis",
}


_PARSE_SYSTEM = (
    "You are a clinical NLP assistant specialized in LOINC retrieval. "
    "Extract distinct laboratory tests, measurements, and clinical observations from text. "
    "Keep analyte + specimen + timing context together when present. "
    "Return ONLY a JSON array of concise strings and nothing else. "
    'Example: ["serum glucose fasting", "hemoglobin a1c", "urine protein"]'
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


def _normalize_text(raw_text: str) -> str:
    text = raw_text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    for src, dst in _LAY_TERM_MAP.items():
        text = text.replace(src, dst)
    return text


def _heuristic_entities(raw_text: str) -> list[str]:
    text = _normalize_text(raw_text)
    entities: list[str] = []

    key_patterns = [
        r"\b(glucose|hemoglobin a1c|creatinine|urea|sodium|potassium|chloride)\b",
        r"\b(complete blood count|comprehensive metabolic panel|basic metabolic panel|lipid panel|urinalysis)\b",
        r"\b(thyroid stimulating hormone|troponin|bilirubin|ast|alt|alkaline phosphatase)\b",
    ]

    for pattern in key_patterns:
        for match in re.finditer(pattern, text):
            entities.append(match.group(1))

    if "fasting" in text:
        entities = [f"{e} fasting" for e in entities] or ["fasting lab test"]
    if "urine" in text and entities:
        entities = [f"urine {e}" if "urine" not in e else e for e in entities]
    if "serum" in text and entities:
        entities = [f"serum {e}" if "serum" not in e else e for e in entities]

    if not entities:
        entities = [text]

    seen = set()
    unique: list[str] = []
    for entity in entities:
        k = entity.strip().lower()
        if k and k not in seen:
            seen.add(k)
            unique.append(entity.strip())
    return unique


async def parse_entities(raw_text: str) -> list[str]:
    """Extract normalized LOINC-oriented query entities."""
    heuristics = _heuristic_entities(raw_text)

    input_tokens = output_tokens = total_tokens = None
    error_msg: str | None = None
    content: str | None = None
    t0 = time.perf_counter()

    try:
        if _is_gemini:
            prompt = f"{_PARSE_SYSTEM}\n\nUser input:\n{raw_text}"
            gemini_client = _get_gemini_client()
            response = await asyncio.wait_for(
                gemini_client.aio.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                    config={"temperature": 0},
                ),
                timeout=20.0,
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
                        {"role": "system", "content": _PARSE_SYSTEM},
                        {"role": "user", "content": raw_text},
                    ],
                ),
                timeout=20.0,
            )
            content = response.choices[0].message.content.strip()
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
    except asyncio.TimeoutError:
        error_msg = "LLM API timeout (> 20 s)"
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    tracker.record_api_call(
        "preprocessing.py",
        "LLM (parse_entities)",
        time.perf_counter() - t0,
        input_tokens,
        output_tokens,
        total_tokens,
        error=error_msg,
    )

    llm_entities: list[str] = []
    if content is not None:
        try:
            llm_entities = json.loads(_clean_json_response(content))
            if not isinstance(llm_entities, list):
                llm_entities = []
        except (json.JSONDecodeError, ValueError):
            llm_entities = []

    merged = heuristics + [str(e).strip() for e in llm_entities if str(e).strip()]

    seen = set()
    unique: list[str] = []
    for entity in merged:
        k = entity.lower().strip()
        if k and k not in seen:
            seen.add(k)
            unique.append(entity.strip())

    console_logger.info(f"LOINC entity extraction: {unique}")
    return unique or [raw_text]
