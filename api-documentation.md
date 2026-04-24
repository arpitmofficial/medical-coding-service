# Medical Coding API Documentation

## Overview

The Medical Coding API accepts free-text clinical notes and returns suggested codes from one or more models:

- ICD-10
- CPT
- LOINC

Protocol and format:

- Protocol: HTTP
- Payload format: JSON
- Framework: FastAPI

Local docs endpoints:

- `GET /docs`
- `GET /openapi.json`

---

## Base URL

```text
http://localhost:8000
```

---

## Authentication

All prediction endpoints require bearer token authentication.

Header:

```http
Authorization: Bearer YOUR_API_KEY
```

---

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | /health | No | Service health and available models |
| POST | /predict | Yes | Run all registered models |
| POST | /predict/icd10 | Yes | Run ICD-10 model only |
| POST | /predict/cpt | Yes | Run CPT model only |
| POST | /predict/loinc | Yes | Run LOINC model only |

---

## Request Schema

### PredictRequest

```json
{
  "clinical_notes": "Type 2 diabetes with neuropathy and foot ulcer"
}
```

Constraints:

- `clinical_notes`: required string, minimum length 1.

---

## Response Schema

### PredictResponse

```json
{
  "clinical_notes": "Type 2 diabetes with neuropathy and foot ulcer",
  "results": [
    {
      "model": "icd10",
      "codes": [
        {
          "code": "E11.40",
          "description": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
          "confidence": 91,
          "explanation": "High-confidence match"
        }
      ],
      "error": null,
      "elapsed_seconds": 1.42
    }
  ],
  "total_elapsed_seconds": 1.99
}
```

Field definitions:

- `results[].model`: model name (`icd10`, `cpt`, `loinc`).
- `results[].codes`: list of model predictions.
- `results[].error`: null on success, error message on model failure.
- `codes[].confidence`: integer range 0 to 100.

### HealthResponse

```json
{
  "status": "ok",
  "models": ["icd10", "cpt", "loinc"]
}
```

---

## Error Handling

HTTP status codes:

- `200`: Request processed.
- `403`: Missing or invalid API key.
- `422`: Invalid request payload.
- `500`: Server configuration issue.

Important behavior:

- `POST /predict` may return HTTP `200` even if one model fails.
- Always inspect `results[].error` for each model.

Example model-level failure in a successful response:

```json
{
  "model": "cpt",
  "codes": [],
  "error": "RuntimeError: Qdrant connection failed",
  "elapsed_seconds": 0.31
}
```

---

## Model-Specific Notes

- ICD-10: typically returns top 5 predictions.
- CPT: typically returns top 5 predictions.
- LOINC: can return a larger result set (commonly up to 50 in current implementation).

Client guidance:

- Add pagination or top-N controls for LOINC results.

---

## Example Requests

### Predict all models

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clinical_notes":"Type 2 diabetes with neuropathy and foot ulcer"}'
```

### Predict ICD-10 only

```bash
curl -X POST "http://localhost:8000/predict/icd10" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clinical_notes":"Acute chest pain radiating to left arm"}'
```

---

## Integration Checklist

1. Validate API key and test `403` behavior.
2. Validate payload schema and test `422` behavior.
3. Use `GET /health` for readiness checks.
4. Parse `results[].error` for per-model failure handling.
5. Handle variable result sizes, especially for LOINC.
