import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_LOINC_APP_DIR = SCRIPT_DIR.parent.parent / "app"
sys.path.insert(0, str(MODEL_LOINC_APP_DIR))

from adaptive_retrieval_loinc import adaptive_retrieve_loinc_candidates

TEST_FILE = SCRIPT_DIR / "LOINC_TestCases.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "eval_summary.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def evaluate() -> None:
    if not TEST_FILE.exists():
        logger.error(f"Test file not found: {TEST_FILE}")
        sys.exit(1)

    df = pd.read_csv(TEST_FILE)
    logger.info(f"Loaded {len(df)} test cases from {TEST_FILE}")

    expected_code_col = "Expected LOINC Code"
    query_col = "Clinical Scenario / Order Description"
    ai_output_col = "AI Output\n(LOINC Code)"
    first_col = "First?"
    top5_col = "in top 5?"

    missing_cols = [
        col
        for col in [expected_code_col, query_col, ai_output_col, first_col, top5_col]
        if col not in df.columns
    ]
    if missing_cols:
        logger.error(f"Missing required column(s): {missing_cols}")
        sys.exit(1)

    # Ensure symbol columns are string-compatible to avoid dtype warnings
    df[ai_output_col] = df[ai_output_col].astype("string")
    df[first_col] = df[first_col].astype("string")
    df[top5_col] = df[top5_col].astype("string")

    correct_first = 0
    correct_top5 = 0
    total = 0

    for idx, row in df.iterrows():
        expected_raw = str(row[expected_code_col]).strip()
        expected_codes = [code.strip() for code in expected_raw.split("/") if code.strip()]

        query = str(row[query_col]).strip()
        if not query:
            continue

        try:
            results = await adaptive_retrieve_loinc_candidates(query)
            output_codes = [str(r.get("code", "")).strip() for r in results[:5] if r.get("code")]
        except Exception as exc:
            logger.error(f"[ERROR] Retrieval failed for row {idx}: {exc}")
            output_codes = []

        df.at[idx, ai_output_col] = " / ".join(output_codes)

        is_first = bool(output_codes) and any(code == output_codes[0] for code in expected_codes)
        in_top5 = any(code in output_codes[:5] for code in expected_codes)

        df.at[idx, first_col] = "" if is_first else ""
        df.at[idx, top5_col] = "" if in_top5 else ""

        correct_first += int(is_first)
        correct_top5 += int(in_top5)
        total += 1

        # Save after each test case so progress is never lost on interruption
        df.to_csv(TEST_FILE, index=False)

    # Final save
    df.to_csv(TEST_FILE, index=False)
    logger.info(f"Results written to {TEST_FILE}")

    first_pct = (100 * correct_first / total) if total else 0.0
    top5_pct = (100 * correct_top5 / total) if total else 0.0

    summary = {
        "total": total,
        "first_correct": correct_first,
        "first_pct": round(first_pct, 2),
        "top5_correct": correct_top5,
        "top5_pct": round(top5_pct, 2),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)
    logger.info(f"Summary saved to {SUMMARY_FILE}")

    print(
        "\nAggregated Stats:\n"
        f"Total: {total}\n"
        f"First correct: {correct_first} ({first_pct:.2f}%)\n"
        f"In top 5: {correct_top5} ({top5_pct:.2f}%)\n"
    )


if __name__ == "__main__":
    asyncio.run(evaluate())
