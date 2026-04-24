# ICD-10 Model

ICD-10 retrieval pipeline used by the medical coding API.

## What It Does

Given free-text clinical notes, this model:

1. Extracts structured medical entities from text.
2. Runs hybrid retrieval against Qdrant (dense + sparse).
3. Fuses candidates with rank-based scoring.
4. Applies LLM reranking.
5. Returns final ICD-10 candidates with confidence and explanation.

## Main Files

- app/adaptive_retrieval.py: Orchestration entrypoint.
- app/preprocessing.py: Clinical entity extraction.
- app/embedding.py: Dense and sparse embedding generation.
- app/qdrant_rest.py: Qdrant search helpers.
- app/reranking.py: Final LLM reranking.
- scripts/query_test.py: Interactive local query script.
- scripts/ingest.py: Ingestion utility for ICD-10 data.

## Expected Environment Variables

Configured via src/.env:

- QDRANT_URL
- QDRANT_API_KEY
- LLM_API_KEY
- LLM_MODEL
- QDRANT_TOP_K (optional)
- FINAL_TOP_N (optional)

## Run Local Query Test

From repository root:

```bash
cd src/model-icd-10/scripts
python query_test.py
```

## Ingest / Rebuild Index

From repository root:

```bash
cd src/model-icd-10/scripts
python ingest.py
```

## Evaluation

Evaluation scripts and outputs are under:

- scripts/Testing

Use this to run reproducible checks on retrieval quality after tuning or data updates.

## API Integration

This model is exposed through the API endpoints:

- POST /predict
- POST /predict/icd10

See the root API docs here:

- ../../api-documentation.md

## Embedding Notes

- Dense embeddings use local SapBERT.
- Sparse embeddings use BM25-style FastEmbed retrieval.
   3. [I25.2] Old myocardial infarction (score: 0.4103)
   ... (30 more)

🤖 LLM RERANKING:
   Original input: 'patient had a heart attack'
   Extracted entity: 'acute myocardial infarction'
   Candidates to evaluate: 33

✅ Found 5 ICD-10 code(s) | Top result: I21.9 (95%)
```

### Final JSON Output

```json
[
  {
    "code": "I21.9",
    "description": "Acute myocardial infarction, unspecified",
    "confidence": 95,
    "explanation": "Direct match for heart attack/acute MI"
  },
  {
    "code": "I23.8",
    "description": "Other current complications following acute myocardial infarction",
    "confidence": 65,
    "explanation": "Related complication code for MI"
  },
  {
    "code": "I21.A1",
    "description": "Myocardial infarction type 2",
    "confidence": 45,
    "explanation": "Specific MI subtype, may apply if type specified"
  },
  {
    "code": "I21.01",
    "description": "ST elevation (STEMI) myocardial infarction of anterior wall",
    "confidence": 45,
    "explanation": "STEMI subtype, applicable if ECG shows ST elevation"
  },
  {
    "code": "I25.2",
    "description": "Old myocardial infarction",
    "confidence": 35,
    "explanation": "History of MI, not current event"
  }
]
```

---

## Troubleshooting

### No Results Returned

| Issue | Solution |
|-------|----------|
| `MIN_SCORE` too high | Set `MIN_SCORE=0.0` in `.env` |
| Entity extraction failed | Check LLM API key and model name |
| Qdrant connection error | Verify `QDRANT_URL` and `QDRANT_API_KEY` |

### Wrong Codes Returned

| Issue | Solution |
|-------|----------|
| Colloquial terms not mapped | Entity extraction should normalize (check `preprocessing.py`) |
| Acronyms not matching | Sparse search should handle (verify BM25 embeddings exist) |
| Semantic drift | Increase `DENSE_SEARCH_LIMIT` for broader semantic pool |

### Performance Issues

| Issue | Solution |
|-------|----------|
| Slow response time | Check network latency to Qdrant/LLM APIs |
| High API costs | Reduce `QDRANT_TOP_K` or use smaller LLM model |
| Memory usage | BM25 model loads lazily, first call is slower |

### Checking Logs

Detailed logs are written to `logs/medical_coding.log`:

```bash
tail -f logs/medical_coding.log
```

---

## Performance Comparison

| Search Type | Semantic Queries | Acronyms | Exact Terms | Overall |
|-------------|-----------------|----------|-------------|---------|
| Dense Only | ✅ Excellent | ❌ Poor | ❌ Poor | Limited |
| Sparse Only | ❌ Poor | ✅ Excellent | ✅ Excellent | Limited |
| **Hybrid (RRF)** | **✅ Excellent** | **✅ Excellent** | **✅ Excellent** | **Best** |

### Real-World Improvements

| Query | Dense Only | Hybrid Result |
|-------|------------|---------------|
| "T2DM" | ❌ Misses diabetes codes | ✅ E11.9 Type 2 diabetes |
| "heart attack" | ✅ I21.9 found | ✅ I21.9 (higher confidence) |
| "CKD stage 3" | ⚠️ Generic CKD | ✅ N18.3 Stage 3 CKD |
| "HTN with CHF" | ❌ Partial match | ✅ I11.0 Hypertensive heart disease |

---

## Database Requirements

### Qdrant Collection: `icd10_hybrid`

- **Records**: ~74,718 ICD-10 codes
- **Vector Types**: Named vectors (`dense` + `sparse`)
- **Payload Fields**: `code`, `description`

### Collection Schema

```json
{
  "collection_name": "icd10_hybrid",
  "vectors_config": {
    "dense": {
      "size": 768,
      "distance": "Cosine"
    },
    "sparse": {}
  }
}
```

---

## License

This project is for educational and research purposes. ICD-10 codes are maintained by WHO and CMS.
