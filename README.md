# Medical Coding Service

Hybrid clinical coding service that predicts ICD-10, CPT, and LOINC codes from free-text clinical notes.

This repository contains:
- A FastAPI gateway in src/api.py
- Three model pipelines under src/model-icd-10, src/model-cpt, and src/model-loinc
- Ingestion and evaluation scripts for each model

## API Documentation

Detailed endpoint documentation is available here:

- [API Documentation](api-documentation.md)

When running the API locally, you can also use:

- http://localhost:8000/docs
- http://localhost:8000/redoc

## Model Documentation

- [ICD-10 Model](src/model-icd-10/README.md)
- [CPT Model](src/model-cpt/README.md)
- [LOINC Model](src/model-loinc/README.md)

## Repository Layout

```text
.
├── README.md
├── api-documentation.md
└── src
    ├── api.py
    ├── requirements.txt
    ├── model-icd-10
    ├── model-cpt
    └── model-loinc
```

## Prerequisites

- Python 3.10+
- Access to a Qdrant instance
- Valid API keys for LLM providers

## Environment Setup

`QDRANT_URL` can point to a local Qdrant instance or to Qdrant Cloud. Use `QDRANT_API_KEY` only when your Qdrant deployment requires authentication.

Create src/.env with at least:

```env
API_KEY=your_api_key

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

LLM_API_KEY=your_llm_api_key
LLM_MODEL=gpt-4o-mini

QDRANT_TOP_K=50
FINAL_TOP_N=5
MIN_SCORE=0.0
```

You can start from the template file:

```bash
cp src/.env.example src/.env
```

### Handoff Notes For New Users

- `API_KEY` is your application auth key for this FastAPI service. It is not generated automatically per user. The team receiving this project should create their own secret value and set it in `src/.env`.
- `QDRANT_URL` and `QDRANT_API_KEY` must come from the new team's own Qdrant instance. They will not get a unique Qdrant URL or key just by running this repo.
- `LLM_API_KEY` must also be their own provider key for the model they choose.
- `.env` is already ignored by git, so the handoff should include instructions, not real secrets.

Example `API_KEY` generation:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Qdrant Setup For Handoff

If you are handing this project to another team, ask them to create their own Qdrant deployment first.

### Option 1: Qdrant Cloud

1. Create an account at Qdrant Cloud.
2. Create a new cluster.
3. Copy the cluster endpoint. This becomes `QDRANT_URL`.
4. Create or copy an API key for that cluster. This becomes `QDRANT_API_KEY`.
5. Put both values in `src/.env`.

Example:

```env
QDRANT_URL=https://your-cluster-id.region.provider.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cloud_api_key
```

### Option 2: Local Qdrant

Run Qdrant locally with Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Then use:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

Supported `LLM_MODEL` values include:

- `gpt-4o-mini`
- `gemini-2.5-flash`
- `llama-3.1-8b-instant`

For `llama-3.1-8b-instant`, use a Groq API key in `LLM_API_KEY`.

## Installation

```bash
cd src
pip install -r requirements.txt
```

## Full Data Ingestion Flow

A new team should do the following in order:

1. Create their own `src/.env` with:
   - `API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `LLM_API_KEY`
   - `LLM_MODEL`
   - A simple way is to copy `src/.env.example` to `src/.env` and fill in their own values
2. Install Python dependencies from `src/`.
3. Make sure the source Excel files are present:
   - `src/model-icd-10/data/icd10cm_codes_2026.xlsx`
   - `src/model-cpt/data/CPTTable.xlsx`
   - `src/model-loinc/data/LOINC_2_82_Active_Codes_Database.xlsx`
4. Run the ingestion scripts to create and populate the Qdrant collections.
5. Start the API only after ingestion finishes successfully.

### Ingestion Commands

From the repository root:

ICD-10:

```bash
cd src/model-icd-10/scripts
python ingest.py
```

This creates and fills the `icd10_hybrid` collection.

CPT:

```bash
cd src/model-cpt/scripts
python ingest.py --recreate
```

This creates and fills the `cpt_hybrid` collection.

LOINC:

```bash
cd src/model-loinc/scripts
python ingest.py
```

This creates and fills the `loinc_hybrid` collection.

### What The Ingestion Scripts Do

- Create or recreate the target Qdrant collection.
- Generate dense embeddings and sparse embeddings for each code record.
- Upload points into Qdrant in batches.
- Store the code metadata used later by the retrieval pipelines.

### After Ingestion

You can verify the setup in this order:

1. Start the API:

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

2. Check health:

```bash
curl http://localhost:8000/health
```

3. Run a smoke test:

```bash
curl -X POST "http://localhost:8000/predict/icd10" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clinical_notes":"Patient presents with acute chest pain radiating to left arm"}'
```

## Run the API

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Smoke Test

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clinical_notes":"Type 2 diabetes with neuropathy and foot ulcer"}'
```

## Client Testing Guide

Use these steps to verify the full system end to end.

1. Start the API server from `src/`.

  ```bash
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
  ```

2. Confirm the service is up.

  ```bash
  curl http://localhost:8000/health
  ```

  Expected result: `status: ok` and the list of available models.

3. Open the interactive API docs in a browser.

  - `http://localhost:8000/docs`
  - `http://localhost:8000/redoc`

4. Test the all-model route.

  ```bash
  curl -X POST "http://localhost:8000/predict" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"clinical_notes":"Type 2 diabetes with neuropathy and foot ulcer"}'
  ```

  Check that the response contains a `results` array and a `total_elapsed_seconds` value.

5. Test each model individually.

  ICD-10:

  ```bash
  curl -X POST "http://localhost:8000/predict/icd10" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"clinical_notes":"Patient presents with acute chest pain radiating to left arm"}'
  ```

  CPT:

  ```bash
  curl -X POST "http://localhost:8000/predict/cpt" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"clinical_notes":"Laparoscopic appendectomy with anesthesia and postoperative care"}'
  ```

  LOINC:

  ```bash
  curl -X POST "http://localhost:8000/predict/loinc" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"clinical_notes":"Serum glucose fasting"}'
  ```

6. Inspect the response format.

  Confirm each model result includes:

  - `model`
  - `codes`
  - `error` should be `null` when the model succeeds
  - `elapsed_seconds`

7. If you are testing from a separate client or staging environment, update the base URL in the commands above to the deployed API host, then repeat the same tests.

8. For deeper model-level validation, run the local scripts directly.

  - `src/model-icd-10/scripts/query_test.py`
  - `src/model-cpt/scripts/query_test.py`
  - `src/model-loinc/scripts/query_test.py`

## Current Pipeline Notes

- ICD-10: hybrid retrieval and reranking enabled with SapBERT dense embeddings.
- CPT: hybrid retrieval and reranking enabled with SapBERT dense embeddings.
- LOINC: retrieval pipeline active; reranking module exists but is currently disabled in the active orchestration path.

## Evaluation Scripts

Representative scripts:

- src/model-icd-10/scripts/query_test.py
- src/model-cpt/scripts/query_test.py
- src/model-loinc/scripts/query_test.py

Model-specific evaluation and ingestion details are documented in each model README.

```bash
# Test health endpoint (no auth needed)
curl http://localhost:8000/health

# Run prediction with cURL
curl -X POST http://localhost:8000/predict/icd10 \
  -H "Authorization: Bearer your_secret_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_notes": "Patient presents with acute chest pain radiating to left arm"
  }'
```

### **Using Python Requests Library**

```python
import requests
import json

# API Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "your_secret_api_key_here"

# Headers with authentication
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Clinical notes to analyze
payload = {
    "clinical_notes": "Patient presents with acute chest pain radiating to left arm"
}

# Make request
response = requests.post(f"{BASE_URL}/predict/icd10", json=payload, headers=headers)

# Print results
print(json.dumps(response.json(), indent=2))
```

### **Using JavaScript/Fetch**

```javascript
const BASE_URL = "http://localhost:8000";
const API_KEY = "your_secret_api_key_here";

const payload = {
  clinical_notes: "Patient presents with acute chest pain radiating to left arm"
};

const headers = {
  "Authorization": `Bearer ${API_KEY}`,
  "Content-Type": "application/json"
};

fetch(`${BASE_URL}/predict/icd10`, {
  method: "POST",
  headers: headers,
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => console.log(JSON.stringify(data, null, 2)))
  .catch(error => console.error("Error:", error));
```

---

## Project Structure

```
medical-coding-service/
├── README.md                          # This file
├── src/
│   ├── api.py                        # Main FastAPI application
│   ├── requirements.txt               # API dependencies
│   ├── .env                          # Environment variables (create this)
│   ├── model-icd-10/                 # ICD-10 model pipeline
│   │   ├── README.md                 # ICD-10 documentation
│   │   ├── requirements.txt          # Model dependencies
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # Configuration & environment setup
│   │   │   ├── adaptive_retrieval.py # Main pipeline entry point
│   │   │   ├── embedding.py          # Vector embeddings
│   │   │   ├── retrieval.py          # Qdrant retrieval
│   │   │   ├── reranking.py          # LLM re-ranking
│   │   │   ├── preprocessing.py      # Text preprocessing
│   │   │   └── ...
│   │   ├── scripts/
│   │   │   ├── ingest.py            # Data ingestion
│   │   │   └── query_test.py        # Query testing
│   │   └── logs/                     # Model logs
│   └── scripts/
│       └── logs/
```

---

### **View Logs**

```bash
# Real-time log monitoring from src/ directory
tail -f logs/medical_coding.log

# Or view older logs
cat logs/medical_coding.log
```

---

## Development Notes

### **Reload on Code Changes**

The `--reload` flag automatically restarts the server when you modify `api.py` or related files.

```bash
uvicorn api:app --reload
```

### **Production Deployment**

For production, use a production ASGI server:

```bash
# Using Gunicorn + Uvicorn workers
gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or using just uvicorn without reload
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Support

For issues or questions:
1. Check logs: `tail -f src/logs/medical_coding.log`
2. Review API documentation: `http://localhost:8000/docs`
3. Verify environment configuration: Check `.env` file setup
