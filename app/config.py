import os
from dotenv import load_dotenv

load_dotenv()

# --- Jina Embeddings ---
JINA_API_KEY = os.getenv("JINA_API_KEY")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HEADERS = {
    "api-key": QDRANT_API_KEY,
    "Content-Type": "application/json",
}

# Keep backward-compatible alias
HEADERS = QDRANT_HEADERS

# --- LLM (OpenAI-compatible) ---
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Pipeline thresholds (adjustable without touching logic) ---
QDRANT_TOP_K: int = int(os.getenv("QDRANT_TOP_K", "50"))   # candidates per entity
FINAL_TOP_N: int = int(os.getenv("FINAL_TOP_N", "5"))       # codes returned to caller
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.80"))    # minimum similarity score