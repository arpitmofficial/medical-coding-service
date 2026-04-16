import argparse
import asyncio
import importlib
import logging
import sys
import uuid
from pathlib import Path

import httpx
import pandas as pd


# =========================
# 📁 PATH SETUP
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CPT_DIR = SCRIPT_DIR.parent
MODEL_ICD10_DIR = MODEL_CPT_DIR.parent / "model-icd-10"

sys.path.insert(0, str(MODEL_ICD10_DIR))

_config = importlib.import_module("app.config")
_embedding = importlib.import_module("app.embedding")

QDRANT_URL = _config.QDRANT_URL
QDRANT_HEADERS = _config.QDRANT_HEADERS
get_embeddings_batch = _embedding.get_embeddings_batch
get_sparse_embeddings_batch = _embedding.get_sparse_embeddings_batch


# =========================
# ⚙️ CONFIG
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "cpt_hybrid"
DEFAULT_PATH = MODEL_CPT_DIR / "data" / "CPTTable.xlsx"
DEFAULT_BATCH = 100


# =========================
# 🧠 COLUMN DETECTION
# =========================
CODE_CANDIDATES = [
    "CPT", "CPT Code", "CPT Codes", "Code", "Procedure Code"
]

DESC_CANDIDATES = [
    "Description",
    "Procedure Description",
    "Procedure Code Description",
    "Procedure Code Descriptions",
    "Long Description",
    "Short Description",
]

CATEGORY_CANDIDATES = [
    "Procedure Code Category",
    "Category",
    "Section"
]


def normalize(col):
    return "".join(c.lower() for c in col if c.isalnum())


def resolve(df, candidates, label):
    mapping = {normalize(c): c for c in df.columns}
    for c in candidates:
        if normalize(c) in mapping:
            return mapping[normalize(c)]
    raise ValueError(f"{label} column not found. Available: {list(df.columns)}")


def clean_code(x):
    x = str(x).strip()
    if x.endswith(".0") and x[:-2].isdigit():
        return x[:-2]
    return x


# =========================
# 🧱 QDRANT SETUP
# =========================
async def setup(client, collection, recreate):
    if recreate:
        logger.info(f"Recreating collection: {collection}")
        await client.delete(f"{QDRANT_URL}/collections/{collection}", headers=QDRANT_HEADERS)

    check = await client.get(f"{QDRANT_URL}/collections/{collection}", headers=QDRANT_HEADERS)
    if check.status_code == 200:
        logger.info("Collection exists, skipping creation")
        return

    schema = {
        "vectors": {
            "dense": {"size": 768, "distance": "Cosine"}
        },
        "sparse_vectors": {
            "sparse": {}
        }
    }

    res = await client.put(
        f"{QDRANT_URL}/collections/{collection}",
        headers=QDRANT_HEADERS,
        json=schema
    )
    res.raise_for_status()
    logger.info("✅ Collection ready")


# =========================
# 📥 INGEST
# =========================
async def ingest(path, collection, batch_size, start, recreate):
    df = pd.read_excel(path)

    code_col = resolve(df, CODE_CANDIDATES, "code")
    desc_col = resolve(df, DESC_CANDIDATES, "description")
    cat_col = resolve(df, CATEGORY_CANDIDATES, "category")

    logger.info(f"Columns → code: {code_col}, desc: {desc_col}, category: {cat_col}")

    df = df.dropna(subset=[code_col, desc_col])
    total = len(df)

    async with httpx.AsyncClient(timeout=120.0) as client:
        await setup(client, collection, recreate)

        uploaded = start

        for i in range(start, total, batch_size):
            batch = df.iloc[i:i + batch_size]

            codes = [clean_code(x) for x in batch[code_col]]
            descs = batch[desc_col].astype(str).str.strip().tolist()
            cats = batch[cat_col].astype(str).str.strip().tolist()

            # 🔥 BEST TEXT FORMAT (hybrid optimized)
            texts = [
                f"CPT {codes[j]} {cats[j]} {descs[j]}"
                for j in range(len(codes))
            ]

            logger.info(f"Embedding {i} → {i + len(texts)}")

            dense = await get_embeddings_batch(texts)
            sparse = await get_sparse_embeddings_batch(texts)

            points = []
            for j in range(len(codes)):
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"cpt::{codes[j]}"))

                points.append({
                    "id": pid,
                    "payload": {
                        "code": codes[j],
                        "description": descs[j],
                        "category": cats[j],
                        "source": "cpt"
                    },
                    "vector": {
                        "dense": dense[j],
                        "sparse": sparse[j]
                    }
                })

            res = await client.put(
                f"{QDRANT_URL}/collections/{collection}/points?wait=true",
                headers=QDRANT_HEADERS,
                json={"points": points}
            )
            res.raise_for_status()

            uploaded += len(points)
            logger.info(f"✅ Uploaded {uploaded}/{total}")

    logger.info("🎉 CPT ingestion complete!")


# =========================
# 🏁 CLI
# =========================
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=str(DEFAULT_PATH))
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--recreate", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse()

    asyncio.run(
        ingest(
            path=Path(args.path),
            collection=args.collection,
            batch_size=args.batch,
            start=args.start,
            recreate=args.recreate,
        )
    )