import os
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

HEADERS = {
    "api-key": QDRANT_API_KEY,
    "Content-Type": "application/json"
}