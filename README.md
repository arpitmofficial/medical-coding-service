# Medical Coding Service

A FastAPI-based medical coding system that retrieves standardized medical codes (ICD-10, CPT, LOINC) from clinical notes using hybrid search with LLM-powered entity extraction and re-ranking.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Documentation & Testing](#api-documentation--testing)
- [API Endpoints](#api-endpoints)
- [Example Usage](#example-usage)

---

## Overview

The Medical Coding Service is a shared gateway that:
- Sits above individual model directories (model-icd-10, model-cpt, model-loinc, etc.)
- Provides a single HTTP interface for doctors to submit clinical diagnoses
- Returns standardized medical codes with confidence scores and clinical explanations

**Current Features:**
- ✅ ICD-10 medical code retrieval
- 🔄 CPT and LOINC endpoints (future implementation)

---

## System Requirements

- **Python 3.9+**
- **pip** or **conda** (Python package manager)
- API keys for external services (see [Environment Variables](#environment-variables))
- Qdrant vector database instance (running or accessible via URL)

---

## Environment Variables

Create a `.env` file in the `src/` directory with the following variables:

### **Required Variables**

```bash
# API Authentication
API_KEY=your_secret_api_key_here

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333          # Qdrant instance URL
QDRANT_API_KEY=your_qdrant_api_key

# Embeddings API
JINA_API_KEY=your_jina_api_key

# LLM Configuration (OpenAI-compatible)
LLM_API_KEY=your_llm_api_key              # e.g., OpenAI or compatible service
LLM_MODEL=gpt-4o-mini                     # LLM model name (default: gpt-4o-mini)
```

### **Optional Variables** (Pipeline Tuning)

```bash
# Pipeline Configuration
QDRANT_TOP_K=50                           # Candidates per entity (default: 50)
FINAL_TOP_N=5                             # Codes returned to caller (default: 5)
MIN_SCORE=0.0                             # Score cutoff threshold (default: 0.0, no filter)
```

### **Example `.env` File**

```bash
cat > src/.env << 'EOF'
API_KEY=medical-api-key-12345
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=qdrant-key-xyz
JINA_API_KEY=jina-embedding-key
LLM_API_KEY=sk-your-openai-key
LLM_MODEL=gpt-4o-mini
QDRANT_TOP_K=50
FINAL_TOP_N=5
MIN_SCORE=0.0
EOF
```

---

## Installation

### 1. **Clone or Navigate to Project**

```bash
cd /path/to/medical-coding-service
```

### 2. **Create a Virtual Environment** (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n medical-coding python=3.9
conda activate medical-coding
```

### 3. **Install Dependencies**

```bash
cd src/

# Install main API dependencies
pip install -r requirements.txt

# Install model-specific dependencies (ICD-10)
pip install -r model-icd-10/requirements.txt
```

### 4. **Verify Installation**

```bash
python -c "import fastapi; import uvicorn; print('✅ All dependencies installed')"
```

---

## Running the API

### **Start the API Server**

From the `src/` directory:

```bash
cd src/
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Command Breakdown:**
- `uvicorn api:app` — Run the FastAPI app from `api.py`
- `--host 0.0.0.0` — Accept connections from any IP
- `--port 8000` — Listen on port 8000 (change as needed)
- `--reload` — Auto-reload on code changes (use only in development)

### **Server Output**

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
INFO:     Registering models …
INFO:     ICD-10 model registered
```

The API is now running and ready to receive requests!

### **Optional: Custom Port**

```bash
uvicorn api:app --host 0.0.0.0 --port 5000
```

Access API at: `http://localhost:5000`

---

## API Documentation & Testing

### **Interactive API Documentation (Swagger UI)**

Once the server is running, visit:

```
http://localhost:8000/docs
```

This provides:
- ✅ Interactive API exploration
- ✅ Live request/response testing
- ✅ Automatic schema validation
- ✅ Beautiful web interface

### **Alternative: ReDoc**

```
http://localhost:8000/redoc
```

Provides detailed API documentation in a different format.

---

## API Endpoints

### **1. Health Check**

**Endpoint:** `GET /health`

**Description:** Check API status and available models

**No Authentication Required**

**Response:**
```json
{
  "status": "ok",
  "models": ["icd10"]
}
```

**Example using cURL:**
```bash
curl http://localhost:8000/health
```

---

### **2. Predict (All Models)**

**Endpoint:** `POST /predict`

**Description:** Run all registered models on clinical notes

**Authentication Required:** `Authorization: Bearer <API_KEY>`

**Request Body:**
```json
{
  "clinical_notes": "Patient presents with acute chest pain radiating to left arm"
}
```

**Response:**
```json
{
  "clinical_notes": "Patient presents with acute chest pain radiating to left arm",
  "results": [
    {
      "model": "icd10",
      "codes": [
        {
          "code": "I21.9",
          "description": "Acute myocardial infarction, unspecified",
          "confidence": 95,
          "explanation": "Acute chest pain with radiation pattern consistent with MI"
        }
      ],
      "error": null,
      "elapsed_seconds": 2.345
    }
  ],
  "total_elapsed_seconds": 2.345
}
```

---

### **3. Predict ICD-10 (Specific Model)**

**Endpoint:** `POST /predict/icd10`

**Description:** Run only the ICD-10 model pipeline

**Authentication Required:** `Authorization: Bearer <API_KEY>`

**Request Body:**
```json
{
  "clinical_notes": "Patient presents with acute chest pain radiating to left arm"
}
```

**Response Format:** Same as `/predict` endpoint

---

## Example Usage

### **Using FastAPI Swagger UI (Browser)**

1. **Start the server:**
   ```bash
   cd src/
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Open browser:**
   ```
   http://localhost:8000/docs
   ```

3. **Test endpoint:**
   - Click on `POST /predict`
   - Click **Try it out**
   - Add your API key in the Authorization field
   - Enter clinical notes in the request body
   - Click **Execute**

### **Using cURL (Command Line)**

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

