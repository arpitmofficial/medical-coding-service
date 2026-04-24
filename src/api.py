"""
Medical Coding API — Shared Gateway
====================================

A FastAPI application that sits **above** individual model directories
(model-icd-10, model-cpt, model-loinc, …) and provides a single HTTP
interface for doctors to submit clinical diagnoses.

Endpoints
---------
POST /predict          — Run all registered models and return combined results.
POST /predict/icd10    — Run only the ICD-10 pipeline.
GET  /health           — Liveness / readiness check.

Authentication
--------------
Every request must include an ``Authorization: Bearer <API_KEY>`` header.
The key is read from the ``API_KEY`` environment variable.

Running
-------
    cd src/
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & path setup
# ---------------------------------------------------------------------------

# Load .env from the src/ directory (where this file lives)
_src_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_src_dir, ".env"))

# So that `from model-icd-10.app import …` style imports are not needed,
# we add individual model directories to sys.path on demand.
_MODEL_ICD10_DIR = os.path.join(_src_dir, "model-icd-10")
_MODEL_CPT_DIR = os.path.join(_src_dir, "model-cpt")
_MODEL_LOINC_DIR = os.path.join(_src_dir, "model-loinc")

API_KEY = os.getenv("API_KEY", "")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Payload sent by the doctor / client system."""
    clinical_notes: str = Field(
        ...,
        min_length=1,
        description="Free-text clinical diagnosis or notes from the doctor.",
        examples=["Patient presents with acute chest pain radiating to left arm"],
    )


class CodeResult(BaseModel):
    code: str = Field(..., description="Medical code (e.g. ICD-10, CPT, LOINC)")
    description: str = Field(..., description="Human-readable code description")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")
    explanation: str = Field(..., description="Brief clinical rationale")


class ModelResult(BaseModel):
    model: str = Field(..., description="Model name, e.g. 'icd10', 'cpt', 'loinc'")
    codes: list[CodeResult] = Field(default_factory=list)
    error: str | None = Field(None, description="Error message if model failed")
    elapsed_seconds: float = Field(..., description="Wall-clock time for this model")


class PredictResponse(BaseModel):
    clinical_notes: str
    results: list[ModelResult]
    total_elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str = "ok"
    models: list[str] = Field(default_factory=list, description="Available models")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_bearer = HTTPBearer()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> str:
    """Validate the Bearer token against the configured API_KEY."""
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: API_KEY not set.",
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return credentials.credentials


# ---------------------------------------------------------------------------
# Model registry — each model exposes an async callable
# ---------------------------------------------------------------------------

# Maps model name → async function(clinical_notes: str) -> list[dict]
_model_registry: dict[str, Any] = {}


def _register_icd10() -> None:
    """Lazily import and register the ICD-10 model."""
    if "icd10" in _model_registry:
        return
    # Add model-icd-10 to sys.path so its internal `from app.…` imports work
    if _MODEL_ICD10_DIR not in sys.path:
        sys.path.insert(0, _MODEL_ICD10_DIR)
    # Now we can import the pipeline entry point
    from app.adaptive_retrieval import adaptive_retrieve_icd_candidates  # noqa: E402
    _model_registry["icd10"] = adaptive_retrieve_icd_candidates

def _register_cpt() -> None:
    """Lazily import and register the CPT model."""
    if "cpt" in _model_registry:
        return
    if _MODEL_CPT_DIR not in sys.path:
        sys.path.insert(0, _MODEL_CPT_DIR)
    from app.adaptive_retrieval_cpt import adaptive_retrieve_cpt_candidates  # noqa: E402
    _model_registry["cpt"] = adaptive_retrieve_cpt_candidates


def _register_loinc() -> None:
    """Lazily import and register the LOINC model."""
    if "loinc" in _model_registry:
        return
    if _MODEL_LOINC_DIR not in sys.path:
        sys.path.insert(0, _MODEL_LOINC_DIR)
    from app.adaptive_retrieval_loinc import adaptive_retrieve_loinc_candidates  # noqa: E402
    _model_registry["loinc"] = adaptive_retrieve_loinc_candidates


# ---------------------------------------------------------------------------
# Lifespan: register models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown hook."""
    logger.info("Registering models …")
    try:
        _register_icd10()
        logger.info("ICD-10 model registered")
    except Exception as exc:
        logger.error("Failed to register ICD-10 model: %s", exc)
        
    try:
        _register_cpt()
        logger.info("CPT model registered")
    except Exception as exc:
        logger.error("Failed to register CPT model: %s", exc)

    try:
        _register_loinc()
        logger.info("LOINC model registered")
    except Exception as exc:
        logger.error("Failed to register LOINC model: %s", exc)
    yield
    logger.info("Shutting down API")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Coding API",
    version="1.0.0",
    description=(
        "Accepts clinical diagnoses from doctors and returns standardised "
        "medical codes (ICD-10, CPT, LOINC)."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Liveness / readiness probe."""
    return HealthResponse(
        status="ok",
        models=list(_model_registry.keys()),
    )


async def _run_model(model_name: str, clinical_notes: str) -> ModelResult:
    """Run a single model and wrap the result (or error) in ModelResult."""
    fn = _model_registry.get(model_name)
    if fn is None:
        return ModelResult(
            model=model_name,
            codes=[],
            error=f"Model '{model_name}' is not registered / available.",
            elapsed_seconds=0.0,
        )

    t0 = time.perf_counter()
    try:
        raw: list[dict[str, Any]] = await fn(clinical_notes)
        elapsed = time.perf_counter() - t0
        codes = [
            CodeResult(
                code=r["code"],
                description=r["description"],
                confidence=r.get("confidence", 0),
                explanation=r.get("explanation", ""),
            )
            for r in raw
        ]
        return ModelResult(model=model_name, codes=codes, elapsed_seconds=round(elapsed, 3))
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception("Model %s failed", model_name)
        return ModelResult(
            model=model_name,
            codes=[],
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=round(elapsed, 3),
        )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_all(
    body: PredictRequest,
    _key: str = Depends(verify_api_key),
):
    """
    Run **all** registered models on the clinical notes and return combined
    results.
    """
    t0 = time.perf_counter()
    model_results: list[ModelResult] = []

    for model_name in _model_registry:
        result = await _run_model(model_name, body.clinical_notes)
        model_results.append(result)

    total = time.perf_counter() - t0
    return PredictResponse(
        clinical_notes=body.clinical_notes,
        results=model_results,
        total_elapsed_seconds=round(total, 3),
    )


@app.post("/predict/icd10", response_model=PredictResponse, tags=["Prediction"])
async def predict_icd10(
    body: PredictRequest,
    _key: str = Depends(verify_api_key),
):
    """Run only the ICD-10 model."""
    t0 = time.perf_counter()
    result = await _run_model("icd10", body.clinical_notes)
    total = time.perf_counter() - t0
    return PredictResponse(
        clinical_notes=body.clinical_notes,
        results=[result],
        total_elapsed_seconds=round(total, 3),
    )


# Future endpoints — uncomment when models are ready

@app.post("/predict/cpt", response_model=PredictResponse, tags=["Prediction"])
async def predict_cpt(body: PredictRequest, _key: str = Depends(verify_api_key)):
    """Run only the CPT model."""
    t0 = time.perf_counter()
    result = await _run_model("cpt", body.clinical_notes)
    total = time.perf_counter() - t0
    return PredictResponse(
        clinical_notes=body.clinical_notes,
        results=[result],
        total_elapsed_seconds=round(total, 3),
    )

@app.post("/predict/loinc", response_model=PredictResponse, tags=["Prediction"])
async def predict_loinc(body: PredictRequest, _key: str = Depends(verify_api_key)):
    """Run only the LOINC model."""
    t0 = time.perf_counter()
    result = await _run_model("loinc", body.clinical_notes)
    total = time.perf_counter() - t0
    return PredictResponse(
        clinical_notes=body.clinical_notes,
        results=[result],
        total_elapsed_seconds=round(total, 3),
    )
