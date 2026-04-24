import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_LOINC_APP_DIR = SCRIPT_DIR.parent.parent / "app"
MODEL_LOINC_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MODEL_LOINC_APP_DIR))

from adaptive_retrieval_loinc import retrieve_loinc_candidates_before_rerank

TEST_FILE = MODEL_LOINC_DIR / "data" / "LOINC_TestCases.xlsx"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "eval_prerank_results.csv"
SUMMARY_FILE = RESULTS_DIR / "eval_prerank_summary.csv"

TARGET_CASES = 67
TEST_SHEET = "LOINC Test Cases"
HEADER_ROW = 1

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _parse_expected_codes(raw: Any) -> list[str]:
    return [code.strip() for code in str(raw).split("/") if code and code.strip()]


async def evaluate() -> None:
    if not TEST_FILE.exists():
        logger.error(f"Test file not found: {TEST_FILE}")
        sys.exit(1)

    df = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET, header=HEADER_ROW)
    logger.info(f"Loaded {len(df)} rows from {TEST_FILE} [{TEST_SHEET}]")

    expected_code_col = "Expected LOINC Code"
    query_col = "Clinical Scenario / Order Description"
    id_col = "Test Case ID"
    class_col = "Class / Domain"

    missing_cols = [
        col
        for col in [id_col, class_col, expected_code_col, query_col]
        if col not in df.columns
    ]
    if missing_cols:
        logger.error(f"Missing required column(s): {missing_cols}")
        sys.exit(1)

    df = df.dropna(subset=[expected_code_col, query_col]).copy()
    available = len(df)
    if available < TARGET_CASES:
        logger.warning(
            "Requested %d cases but only %d valid cases available; evaluating all available rows.",
            TARGET_CASES,
            available,
        )
        cases_df = df
    else:
        cases_df = df.head(TARGET_CASES)

    logger.info("Evaluating %d LOINC cases without reranking", len(cases_df))

    pass_count = 0
    total = 0
    rows: list[dict[str, Any]] = []

    for _, row in cases_df.iterrows():
        expected_codes = _parse_expected_codes(row[expected_code_col])

        query = str(row[query_col]).strip()
        if not query:
            continue

        try:
            candidates = await retrieve_loinc_candidates_before_rerank(query)
            output_codes = [
                str(r.get("code", "")).strip() for r in candidates if r.get("code")
            ]
        except Exception as exc:
            logger.error(
                "[ERROR] Retrieval failed for case %s: %s", row.get(id_col, "N/A"), exc
            )
            output_codes = []

        first_match_rank = next(
            (i + 1 for i, code in enumerate(output_codes) if code in expected_codes),
            None,
        )
        in_candidate_pool = first_match_rank is not None

        rows.append(
            {
                "Test Case ID": row.get(id_col, ""),
                "Class / Domain": row.get(class_col, ""),
                "Clinical Scenario / Order Description": query,
                "Expected LOINC Code": " / ".join(expected_codes),
                "Candidate Count (Pre-rerank)": len(output_codes),
                "First Match Rank (Pre-rerank)": (
                    first_match_rank if first_match_rank else "N/A"
                ),
                "Pass / Fail": "PASS" if in_candidate_pool else "FAIL",
                "Candidate Codes (Top 50)": " / ".join(output_codes[:50]),
            }
        )

        pass_count += int(in_candidate_pool)
        total += 1

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_FILE, index=False)
    logger.info("Pre-rerank results saved to %s", RESULTS_FILE)

    pass_pct = (100 * pass_count / total) if total else 0.0

    summary = {
        "requested_cases": TARGET_CASES,
        "evaluated_cases": total,
        "pass_count_in_prerank_pool": pass_count,
        "pass_pct_in_prerank_pool": round(pass_pct, 2),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)
    logger.info("Pre-rerank summary saved to %s", SUMMARY_FILE)

    print(
        "\nLOINC Pre-rerank Coverage Stats:\n"
        f"Requested Cases: {TARGET_CASES}\n"
        f"Evaluated Cases: {total}\n"
        f"Expected Code Found in Candidate Pool: {pass_count} ({pass_pct:.2f}%)\n"
    )


if __name__ == "__main__":
    asyncio.run(evaluate())
