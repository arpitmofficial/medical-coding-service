
import pandas as pd
import sys
from pathlib import Path
import logging

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_FILE = SCRIPT_DIR / "CPT_TestCases.xlsx"

# Store summary in 'results' subdirectory
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "eval_summary.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


import pandas as pd
import sys
from pathlib import Path
import logging
import asyncio
import time

# Setup paths for imports


SCRIPT_DIR = Path(__file__).resolve().parent

import pandas as pd
import sys
from pathlib import Path
import logging
import asyncio

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CPT_DIR = SCRIPT_DIR.parent.parent / "app"
sys.path.insert(0, str(MODEL_CPT_DIR))
from adaptive_retrieval_cpt import adaptive_retrieve_cpt_candidates

TEST_FILE = SCRIPT_DIR / "CPT_TestCases.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "eval_summary.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

async def evaluate():
    if not TEST_FILE.exists():
        logger.error(f"Test file not found: {TEST_FILE}")
        sys.exit(1)

    df = pd.read_csv(TEST_FILE)
    logger.info(f"Loaded {len(df)} test cases from {TEST_FILE}")

    expected_code_col = "Expected CPT Code"
    ai_output_col = "AI Output\n(CPT Code)"
    first_col = "First?"
    top5_col = "in top 5?"

    correct_first = 0
    correct_top5 = 0
    total = 0

    for idx, row in df.iterrows():
        expected_raw = str(row[expected_code_col]).strip()
        # Support multiple expected codes separated by slashes
        expected_codes = [e.strip() for e in expected_raw.split("/") if e.strip()]
        query = str(row["Clinical Scenario / Operative Note Summary"]).strip()
        if not query:
            continue
        try:
            results = await adaptive_retrieve_cpt_candidates(query)
        except Exception as exc:
            logger.error(f"[ERROR] Retrieval failed for row {idx}: {exc}")
            df.at[idx, ai_output_col] = ""
            df.at[idx, first_col] = ""
            df.at[idx, top5_col] = ""
            total += 1
            df.to_csv(TEST_FILE, index=False)  # Save after each row
            continue
        # Get top 5 codes from results
        output_codes = [r["code"] for r in results[:5] if "code" in r]
        df.at[idx, ai_output_col] = " / ".join(output_codes)
        # Success if any expected code matches
        is_first = (output_codes and any(ec == output_codes[0] for ec in expected_codes))
        in_top5 = any(ec in output_codes[:5] for ec in expected_codes)
        df.at[idx, first_col] = "" if is_first else ""
        df.at[idx, top5_col] = "" if in_top5 else ""
        if is_first:
            correct_first += 1
        if in_top5:
            correct_top5 += 1
        total += 1
        df.to_csv(TEST_FILE, index=False)  # Save after each row

    df.to_csv(TEST_FILE, index=False)
    logger.info(f"Results written to {TEST_FILE}")

    first_pct = 100 * correct_first / total if total else 0.0
    top5_pct = 100 * correct_top5 / total if total else 0.0
    summary = {
        "total": total,
        "first_correct": correct_first,
        "first_pct": round(first_pct, 2),
        "top5_correct": correct_top5,
        "top5_pct": round(top5_pct, 2)
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)
    logger.info(f"Summary saved to {SUMMARY_FILE}")
    print(f"\nAggregated Stats:\nTotal: {total}\nFirst correct: {correct_first} ({first_pct:.2f}%)\nIn top 5: {correct_top5} ({top5_pct:.2f}%)\n")

if __name__ == "__main__":
    asyncio.run(evaluate())
