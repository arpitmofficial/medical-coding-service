import asyncio
import httpx
import logging
import os
import sys
import pandas as pd
import uuid
import math

# =========================
#  FIX IMPORT PATH (model-icd-10/app)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_ICD10_DIR = os.path.join(BASE_DIR, "..", "model-icd-10")
sys.path.insert(0, MODEL_ICD10_DIR)

from app.config import QDRANT_URL, QDRANT_HEADERS
from app.embedding import get_embeddings_batch, get_sparse_embeddings_batch

# =========================
#  CONFIG
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "loinc_hybrid"

EXCEL_FILE_PATH = os.path.join(
    BASE_DIR, "data", "LOINC_2_82_Active_Codes_Database.xlsx"
)

BATCH_SIZE = 100


# =========================
#  HELPERS
# =========================
def clean_value(v):
    return None if pd.isna(v) else v


def clean_vector(vec):
    return [
        0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v) for v in vec
    ]


def normalize(value):
    if pd.isna(value):
        return ""
    return str(value).replace("^", "").lower().strip()


# =========================
#  TEXT BUILDER
# =========================
def build_text(row):
    code = normalize(row.get("LOINC Code"))
    component = normalize(row.get("Component (What is measured)"))
    system = normalize(row.get("System (Specimen/Source)"))
    prop = normalize(row.get("Property"))
    time_aspect = normalize(row.get("Time Aspect"))
    scale = normalize(row.get("Scale Type"))
    method = normalize(row.get("Method Type"))
    class_name = normalize(row.get("Class"))
    long_name = normalize(row.get("Long Common Name"))
    short_name = normalize(row.get("Short Name"))

    # Field labels improve retrieval by preserving axis semantics (component/system/time/etc).
    labeled_parts = [
        f"loinc code {code}" if code else "",
        f"component {component}" if component else "",
        f"system {system}" if system else "",
        f"property {prop}" if prop else "",
        f"time {time_aspect}" if time_aspect else "",
        f"scale {scale}" if scale else "",
        f"method {method}" if method else "",
        f"class {class_name}" if class_name else "",
        f"long name {long_name}" if long_name else "",
        f"short name {short_name}" if short_name else "",
    ]

    # Mild weighting for key discriminative facets.
    weighted = [
        component,
        component,
        system,
        system,
        prop,
        time_aspect,
        long_name,
        short_name,
    ]

    return " ".join([p for p in labeled_parts + weighted if p])


# =========================
#  SETUP COLLECTION
# =========================
async def setup_collection(client):
    logger.info(f"Recreating collection: {COLLECTION_NAME}")

    await client.delete(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}", headers=QDRANT_HEADERS
    )

    schema = {
        "vectors": {"dense": {"size": 768, "distance": "Cosine"}},
        "sparse_vectors": {"sparse": {}},
    }

    res = await client.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        headers=QDRANT_HEADERS,
        json=schema,
    )
    res.raise_for_status()

    logger.info(" Collection ready")


# =========================
#  INGEST
# =========================
async def ingest():
    logger.info(f"Reading file: {EXCEL_FILE_PATH}")

    df = pd.read_excel(EXCEL_FILE_PATH)
    df = df.dropna(subset=["LOINC Code", "Component (What is measured)"])

    total = len(df)
    logger.info(f"Total records: {total}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        await setup_collection(client)

        uploaded = 0

        for start in range(0, total, BATCH_SIZE):
            batch = df.iloc[start : start + BATCH_SIZE]

            texts, payloads, ids = [], [], []

            for _, row in batch.iterrows():
                code = str(row["LOINC Code"]).strip()

                text = build_text(row)
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"loinc::{code}"))

                payload = {
                    "code": code,
                    "component": clean_value(row.get("Component (What is measured)")),
                    "property": clean_value(row.get("Property")),
                    "time": clean_value(row.get("Time Aspect")),
                    "system": clean_value(row.get("System (Specimen/Source)")),
                    "scale": clean_value(row.get("Scale Type")),
                    "method": clean_value(row.get("Method Type")),
                    "class": clean_value(row.get("Class")),
                    "class_type": clean_value(row.get("Class Type")),
                    "long_name": clean_value(row.get("Long Common Name")),
                    "short_name": clean_value(row.get("Short Name")),
                    "status": clean_value(row.get("Status")),
                }

                texts.append(text)
                payloads.append(payload)
                ids.append(pid)

            logger.info(f"Embedding {start} → {start + len(texts)}")

            dense = await get_embeddings_batch(texts)
            sparse = await get_sparse_embeddings_batch(texts)

            points = []
            for i in range(len(texts)):
                points.append(
                    {
                        "id": ids[i],
                        "payload": payloads[i],
                        "vector": {
                            "dense": clean_vector(dense[i]),
                            "sparse": sparse[i],
                        },
                    }
                )

            res = await client.put(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points?wait=true",
                headers=QDRANT_HEADERS,
                json={"points": points},
            )
            res.raise_for_status()

            uploaded += len(points)
            logger.info(f" Uploaded {uploaded}/{total}")

        logger.info(" Done")


# =========================
if __name__ == "__main__":
    asyncio.run(ingest())
