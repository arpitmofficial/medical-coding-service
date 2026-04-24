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
- Valid API keys for embedding and LLM providers

## Environment Setup

Create src/.env with at least:

```env
API_KEY=your_api_key

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

JINA_API_KEY=your_jina_api_key
LLM_API_KEY=your_llm_api_key
LLM_MODEL=gpt-4o-mini
# other llm models that can be used are gemini-1.5-flash and llama-3.1-8b-instant
# for llama model generate groq api key

QDRANT_TOP_K=50
FINAL_TOP_N=5
MIN_SCORE=0.0
```

## Installation

```bash
cd src
pip install -r requirements.txt
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

## Current Pipeline Notes

- ICD-10: hybrid retrieval and reranking enabled.
- CPT: hybrid retrieval and reranking enabled.
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

## Troubleshooting

### **Issue: "API_KEY not set" error**

**Solution:**
- Ensure `.env` file exists in `src/` directory
- Verify `API_KEY` variable is set: `cat src/.env | grep API_KEY`
- Restart the uvicorn server

### **Issue: "Invalid API key" (403 Forbidden)**

**Solution:**
- Ensure the Bearer token in `Authorization` header matches the `API_KEY` in `.env`
- Check header format: `Authorization: Bearer <your_key>`

### **Issue: Qdrant connection error**

**Solution:**
- Verify Qdrant is running: `curl http://localhost:6333/health`
- Check `QDRANT_URL` in `.env` file
- Verify `QDRANT_API_KEY` is correct

### **Issue: Jina API authentication failed**

**Solution:**
- Verify `JINA_API_KEY` is correct and active
- Check network connectivity to Jina embedding service

### **Issue: LLM API errors**

**Solution:**
- Verify `LLM_API_KEY` is valid and has sufficient quota
- Check `LLM_MODEL` name is supported by your LLM provider
- Monitor logs in `src/logs/medical_coding.log`

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

---

## License

[Specify your license here]

---

## Support

For issues or questions:
1. Check logs: `tail -f src/logs/medical_coding.log`
2. Review API documentation: `http://localhost:8000/docs`
3. Verify environment configuration: Check `.env` file setup

