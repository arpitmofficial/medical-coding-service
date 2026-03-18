# ICD-10 Hybrid Search & Retrieval System

A production-ready medical coding system that retrieves ICD-10 codes using **hybrid search** (semantic + keyword) with LLM-powered entity extraction and re-ranking.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Pipeline Flow](#pipeline-flow)
3. [Configuration Values](#configuration-values)
4. [Search Splits & Fusion](#search-splits--fusion)
5. [Component Details](#component-details)
6. [Quick Start](#quick-start)
7. [Example Output](#example-output)
8. [Troubleshooting](#troubleshooting)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ICD-10 HYBRID RETRIEVAL PIPELINE                     │
└─────────────────────────────────────────────────────────────────────────────┘

     USER INPUT                    STAGE 0                     STAGE 1
  ┌─────────────┐            ┌─────────────────┐         ┌─────────────────┐
  │ "patient    │            │  LLM ENTITY     │         │  DUAL EMBEDDING │
  │  had a      │ ────────▶  │  EXTRACTION     │ ──────▶ │                 │
  │  heart      │            │  (Gemini)       │         │ Dense + Sparse  │
  │  attack"    │            └─────────────────┘         └─────────────────┘
  └─────────────┘                    │                          │
                                     ▼                          ▼
                          "acute myocardial           ┌─────────────────────┐
                           infarction"                │ Jina (768D) Vector  │
                                                      │ BM25 Sparse Vector  │
                                                      └─────────────────────┘
                                                               │
                              ┌─────────────────────────────────┘
                              ▼
                    STAGE 2: QDRANT HYBRID SEARCH
     ┌────────────────────────────────────────────────────────────────┐
     │                                                                │
     │  ┌──────────────────┐         ┌──────────────────┐            │
     │  │  DENSE SEARCH    │         │  SPARSE SEARCH   │            │
     │  │  (Semantic)      │         │  (Keyword/BM25)  │            │
     │  │                  │         │                  │            │
     │  │  Top 30 results  │         │  Top 20 results  │            │
     │  │  Cosine distance │         │  BM25 scoring    │            │
     │  └────────┬─────────┘         └────────┬─────────┘            │
     │           │                            │                       │
     │           └──────────┬─────────────────┘                       │
     │                      ▼                                         │
     │           ┌──────────────────────┐                             │
     │           │    RRF FUSION        │                             │
     │           │  (Reciprocal Rank)   │                             │
     │           │                      │                             │
     │           │  Top 50 combined     │                             │
     │           └──────────────────────┘                             │
     │                                                                │
     └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    STAGE 3: MERGE & DEDUPLICATE
     ┌────────────────────────────────────────────────────────────────┐
     │  • Remove duplicate codes (keep highest score)                 │
     │  • Filter by MIN_SCORE threshold (default: 0.0 = no filter)    │
     │  • Sort by RRF score descending                                │
     └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    STAGE 4: LLM RE-RANKING
     ┌────────────────────────────────────────────────────────────────┐
     │                      GEMINI LLM                                │
     │                                                                │
     │  Inputs:                                                       │
     │  • Original user input: "patient had a heart attack"           │
     │  • Extracted entity: "acute myocardial infarction"             │
     │  • All hybrid candidates with similarity scores                │
     │                                                                │
     │  Output:                                                       │
     │  • Exactly 5 codes with confidence scores (0-100%)             │
     │  • Clinical explanation for each code                          │
     └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        FINAL OUTPUT
     ┌────────────────────────────────────────────────────────────────┐
     │  [                                                             │
     │    {"code": "I21.9", "description": "...", "confidence": 95},  │
     │    {"code": "I23.8", "description": "...", "confidence": 65},  │
     │    {"code": "I21.A1", "description": "...", "confidence": 45}, │
     │    {"code": "I21.01", "description": "...", "confidence": 45}, │
     │    {"code": "I25.2", "description": "...", "confidence": 35}   │
     │  ]                                                             │
     └────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

### Stage 0: LLM Entity Extraction
**File**: `app/preprocessing.py`

| Aspect | Detail |
|--------|--------|
| **Purpose** | Convert colloquial/layman terms to standard medical terminology |
| **LLM** | Gemini 2.5-flash (or OpenAI-compatible) |
| **Temperature** | 0 (deterministic) |
| **Output** | JSON array of medical entities |

**Term Normalization Examples**:
| User Input | Extracted Entity |
|------------|------------------|
| "heart attack" | "acute myocardial infarction" |
| "stroke" | "cerebrovascular accident" |
| "high blood pressure" | "hypertension" |
| "sugar disease" | "diabetes mellitus" |
| "broken bone" | "fracture" |

---

### Stage 1: Dual Embedding Generation
**File**: `app/embedding.py`

| Embedding Type | Model | Dimensions | Purpose |
|----------------|-------|------------|---------|
| **Dense** | Jina v2-base-en | 768 | Semantic similarity (understands meaning) |
| **Sparse** | Qdrant/BM25 (FastEmbed) | Variable (sparse) | Keyword matching (exact terms, acronyms) |

**Why Two Embeddings?**
- **Dense only**: Great at "heart attack" → "myocardial infarction" but misses "T2DM" → "Type 2 Diabetes"
- **Sparse only**: Great at acronyms but misses semantic relationships
- **Combined**: Best of both worlds

---

### Stage 2: Qdrant Hybrid Search
**File**: `app/qdrant_rest.py`

#### Search Limits (Hardcoded Constants)

| Search Type | Limit | Constant Name |
|-------------|-------|---------------|
| **Dense (Semantic)** | 30 | `DENSE_SEARCH_LIMIT` |
| **Sparse (BM25/Keyword)** | 20 | `SPARSE_SEARCH_LIMIT` |
| **Hybrid (RRF Combined)** | 50 | `HYBRID_RESULT_LIMIT` |

#### Vector Configuration in Qdrant

```python
{
  "vectors": {
    "dense": {
      "size": 768,           # Jina embedding dimensions
      "distance": "Cosine"   # Similarity metric (0-1 score)
    },
    "sparse": {
      "type": "sparse"       # BM25 sparse vectors
    }
  }
}
```

#### Reciprocal Rank Fusion (RRF)

The RRF algorithm combines results from dense and sparse searches:

```
RRF_score(d) = Σ 1 / (k + rank(d))
```

Where:
- `k` = 60 (Qdrant default constant)
- `rank(d)` = position of document d in each result list
- Higher RRF score = appears highly ranked in both/multiple searches

**Example**:
| Code | Dense Rank | Sparse Rank | RRF Calculation | RRF Score |
|------|------------|-------------|-----------------|-----------|
| I21.9 | 1 | 1 | 1/(60+1) + 1/(60+1) | 0.0328 |
| I23.8 | 4 | 2 | 1/(60+4) + 1/(60+2) | 0.0317 |
| I25.2 | 2 | 12 | 1/(60+2) + 1/(60+12) | 0.0300 |

*Note: Qdrant normalizes RRF scores to 0-1 range where 1.0 = top match*

---

### Stage 3: Merge & Deduplicate
**File**: `app/adaptive_retrieval.py`

| Operation | Detail |
|-----------|--------|
| **Deduplication** | If same ICD code appears multiple times, keep highest score |
| **Score Filtering** | Remove codes below `MIN_SCORE` threshold |
| **Sorting** | Order by RRF score descending |

---

### Stage 4: LLM Re-ranking
**File**: `app/reranking.py`

| Aspect | Detail |
|--------|--------|
| **LLM** | Gemini 2.5-flash |
| **Input** | Original user text + Extracted entity + All candidates |
| **Output** | Exactly 5 codes with confidence scores |

**Confidence Score Guidelines**:
| Range | Meaning |
|-------|---------|
| 90-100% | Perfect semantic match with clinical text |
| 75-89% | Strong match, highly appropriate code |
| 60-74% | Good match, reasonable code choice |
| 40-59% | Moderate match, possible but not ideal |
| <40% | Weak match, only if no better options |

---

## Configuration Values

### Environment Variables (`.env`)

```env
# API Keys
JINA_API_KEY=your_jina_api_key
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
LLM_API_KEY=your_gemini_or_openai_key
LLM_MODEL=gemini-2.5-flash

# Pipeline Configuration
QDRANT_TOP_K=50      # Max candidates per entity (before fusion)
FINAL_TOP_N=5        # Final codes returned (strict top 5)
MIN_SCORE=0.0        # Score cutoff (0.0 = no cutoff)
```

### Hardcoded Constants

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `DENSE_SEARCH_LIMIT` | `qdrant_rest.py` | 30 | Dense search results before fusion |
| `SPARSE_SEARCH_LIMIT` | `qdrant_rest.py` | 20 | Sparse search results before fusion |
| `HYBRID_RESULT_LIMIT` | `qdrant_rest.py` | 50 | Max results after RRF fusion |
| `_JINA_MODEL` | `embedding.py` | "jina-embeddings-v2-base-en" | Dense embedding model |
| Sparse Model | `embedding.py` | "Qdrant/bm25" | BM25 sparse embedding |

---

## Search Splits & Fusion

### Why 30/20/50 Split?

| Ratio | Reasoning |
|-------|-----------|
| **Dense: 30** | Larger pool for semantic matches (catches synonyms, related concepts) |
| **Sparse: 20** | Focused keyword matches (acronyms, exact terms) |
| **Total: 50** | RRF combines both, but overlapping codes boost each other's rank |

### How RRF Fusion Works

1. **Execute both searches independently**:
   - Dense search returns top 30 codes (semantic similarity)
   - Sparse search returns top 20 codes (keyword matching)

2. **Calculate RRF score for each code**:
   - Codes appearing in **both** lists get higher combined scores
   - A code ranked #1 in dense and #1 in sparse = highest possible score
   - A code ranked #30 in dense only = very low score

3. **Return top 50 by RRF score**:
   - Usually fewer than 50 unique codes (due to overlap)
   - Example: 30 + 20 searches might yield ~33 unique codes

### Score Ranges

| Search Type | Score Range | Interpretation |
|-------------|-------------|----------------|
| **Dense (Cosine)** | 0.0 - 1.0 | 1.0 = identical, 0.7+ = strong match |
| **Sparse (BM25)** | 0.0 - ∞ | Unbounded positive, higher = better keyword match |
| **RRF (Normalized)** | 0.0 - 1.0 | 1.0 = top combined rank, drops quickly |

---

## Component Details

### File Structure

```
src/
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment loading, logging setup
│   ├── embedding.py           # Jina dense + BM25 sparse embeddings
│   ├── preprocessing.py       # LLM entity extraction
│   ├── qdrant_rest.py         # Qdrant REST API calls, hybrid search
│   ├── reranking.py           # LLM re-ranking with confidence scores
│   ├── retrieval.py           # Legacy retrieval (if present)
│   └── adaptive_retrieval.py  # Main pipeline orchestration
├── scripts/
│   └── query_test.py          # Test script
├── logs/
│   └── medical_coding.log     # Detailed debug logs
├── .env                       # Environment configuration
└── requirements.txt           # Python dependencies
```

### Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `parse_entities()` | `preprocessing.py` | LLM extracts medical entities from raw text |
| `get_embeddings_batch()` | `embedding.py` | Generate Jina dense vectors |
| `get_sparse_embeddings_batch()` | `embedding.py` | Generate BM25 sparse vectors |
| `search_vectors_debug()` | `qdrant_rest.py` | Execute hybrid search with debug output |
| `rerank_codes()` | `reranking.py` | LLM re-rank candidates to top 5 |
| `adaptive_retrieve_icd_candidates()` | `adaptive_retrieval.py` | Full pipeline orchestration |

---

## Quick Start

### 1. Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file in `src/` directory:

```env
JINA_API_KEY=your_jina_key
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_key
LLM_API_KEY=your_gemini_key
LLM_MODEL=gemini-2.5-flash
QDRANT_TOP_K=50
FINAL_TOP_N=5
MIN_SCORE=0.0
```

### 3. Test the System

```bash
python scripts/query_test.py
```

Or run directly:

```python
import asyncio
from app.adaptive_retrieval import adaptive_retrieve_icd_candidates

async def main():
    results = await adaptive_retrieve_icd_candidates("patient had a heart attack")
    for r in results:
        print(f"{r['code']}: {r['description']} ({r['confidence']}%)")

asyncio.run(main())
```

---

## Example Output

### Terminal Output (Debug Mode)

```
🔍 Processing: 'patient had a heart attack'
🔍 🧠 LLM Entity Extraction: ['acute myocardial infarction']
🔍 Hybrid Search ENABLED (✨ Dense + Sparse vectors with RRF fusion)

📊 Searching for entity: 'acute myocardial infarction'

============================================================
🔵 DENSE (SEMANTIC) SEARCH RESULTS (top 30):
============================================================
   1. [I21.9] Acute myocardial infarction, unspecified (score: 0.9228)
   2. [I25.2] Old myocardial infarction (score: 0.8776)
   3. [I21.A1] Myocardial infarction type 2 (score: 0.8715)
   ... (27 more)

============================================================
🟡 SPARSE (KEYWORD/BM25) SEARCH RESULTS (top 20):
============================================================
   1. [I21.9] Acute myocardial infarction, unspecified (score: 8.4314)
   2. [I23.8] Other current complications following AMI (score: 8.3865)
   3. [I24.0] Acute coronary thrombosis not resulting in MI (score: 8.3865)
   ... (17 more)

============================================================
🟢 HYBRID (RRF FUSION) SEARCH RESULTS (top 50):
============================================================
   1. [I21.9] Acute myocardial infarction, unspecified (score: 1.0000)
   2. [I23.8] Other current complications following AMI (score: 0.5333)
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
