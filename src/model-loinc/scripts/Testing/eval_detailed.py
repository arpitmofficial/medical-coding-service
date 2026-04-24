"""
Detailed Evaluation Script — LOINC No-Rerank Pipeline
======================================================

Runs every test case through the current pipeline (fusion-only, no LLM
reranking) and produces two CSV files:

  results/eval_detailed_results.csv   — one row per test case, all signals
  results/eval_detailed_summary.csv   — overall + per-class aggregates

Metrics captured per case
--------------------------
  hit@1  — correct code is rank 1 in candidate pool
  hit@3  — correct code is in top-3 of candidate pool
  hit@5  — correct code is in top-5 of candidate pool
  hit@10 — correct code is in top-10 of candidate pool
  hit@any — correct code appears anywhere in the candidate pool (recall)
  first_match_rank   — exact rank of the first correct code (or N/A)
  pool_size          — total candidates returned

  For the top-5 returned codes the script records:
    rank, code, confidence, dense_norm, sparse_norm, rrf_norm,
    metadata_bonus, fused_score, explanation snippet

Usage
-----
    cd src/model-loinc/scripts/Testing
    python eval_detailed.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path wiring — same approach as eval.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_LOINC_APP = SCRIPT_DIR.parent.parent / "app"
MODEL_LOINC_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MODEL_LOINC_APP))

from adaptive_retrieval_loinc import (  # noqa: E402
    retrieve_loinc_candidates_before_rerank,
    adaptive_retrieve_loinc_candidates,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_FILE = MODEL_LOINC_DIR / "data" / "LOINC_TestCases.xlsx"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_CSV = RESULTS_DIR / "eval_detailed_results.csv"
SUMMARY_CSV = RESULTS_DIR / "eval_detailed_summary.csv"

TEST_SHEET = "LOINC Test Cases"
HEADER_ROW = 1
TARGET_CASES = 67  # evaluate all available cases up to this limit
TOP_N_FINAL = 5  # how many final codes the pipeline returns

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COL_ID = "Test Case ID"
COL_CLASS = "Class / Domain"
COL_QUERY = "Clinical Scenario / Order Description"
COL_EXP = "Expected LOINC Code"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_expected(raw: Any) -> list[str]:
    return [c.strip() for c in str(raw).split("/") if c.strip()]


def _hit_at_k(output_codes: list[str], expected: list[str], k: int) -> bool:
    return any(c in expected for c in output_codes[:k])


def _first_rank(output_codes: list[str], expected: list[str]) -> int | None:
    for i, c in enumerate(output_codes, start=1):
        if c in expected:
            return i
    return None


def _safe(val: Any, fmt: str = "") -> str:
    if val is None:
        return "N/A"
    if fmt:
        return format(val, fmt)
    return str(val)


# ---------------------------------------------------------------------------
# Core per-case evaluation
# ---------------------------------------------------------------------------


async def _evaluate_case(row: pd.Series) -> dict[str, Any] | None:
    expected_codes = _parse_expected(row[COL_EXP])
    query = str(row[COL_QUERY]).strip()
    case_id = str(row.get(COL_ID, ""))
    class_domain = str(row.get(COL_CLASS, ""))

    if not query:
        return None

    # ── Stage A: full pre-rerank candidate pool (for pool/rank metrics) ──
    try:
        pool: list[dict[str, Any]] = await retrieve_loinc_candidates_before_rerank(
            query
        )
    except Exception as exc:
        logger.error("Retrieval failed for %s: %s", case_id, exc)
        pool = []

    pool_codes = [str(c.get("code", "")).strip() for c in pool if c.get("code")]

    # ── Stage B: final top-N output (pipeline output without reranking) ──
    try:
        final: list[dict[str, Any]] = await adaptive_retrieve_loinc_candidates(query)
    except Exception as exc:
        logger.error("Final retrieve failed for %s: %s", case_id, exc)
        final = []

    final_codes = [str(c.get("code", "")).strip() for c in final if c.get("code")]

    # ── Metrics ──
    pool_size = len(pool_codes)
    first_rank_pool = _first_rank(pool_codes, expected_codes)
    first_rank_final = _first_rank(final_codes, expected_codes)

    hit_any = first_rank_pool is not None
    hit_at_1 = _hit_at_k(pool_codes, expected_codes, 1)
    hit_at_3 = _hit_at_k(pool_codes, expected_codes, 3)
    hit_at_5 = _hit_at_k(pool_codes, expected_codes, 5)
    hit_at_10 = _hit_at_k(pool_codes, expected_codes, 10)

    final_hit = first_rank_final is not None
    pass_fail = "PASS" if hit_any else "FAIL"

    # ── Build top-5 detail columns (from final output) ──
    top5_cols: dict[str, Any] = {}
    for rank in range(1, TOP_N_FINAL + 1):
        if rank <= len(final):
            c = final[rank - 1]
            code = str(c.get("code", "")).strip()
            correct = "✓" if code in expected_codes else ""
            confidence = c.get("confidence", "N/A")
            # explanation carries "Dense: X, Sparse: Y, RRF: Z, Metadata: W"
            expl = str(c.get("explanation", ""))
            desc = str(c.get("description", ""))
        else:
            code = correct = confidence = expl = desc = ""

        pfx = f"top{rank}"
        top5_cols[f"{pfx}_code"] = code
        top5_cols[f"{pfx}_correct"] = correct
        top5_cols[f"{pfx}_confidence"] = confidence
        top5_cols[f"{pfx}_desc"] = desc[:80]  # truncate long names
        top5_cols[f"{pfx}_signals"] = expl

    row_data: dict[str, Any] = {
        COL_ID: case_id,
        COL_CLASS: class_domain,
        COL_QUERY: query,
        "Expected LOINC Code(s)": " / ".join(expected_codes),
        # ── pool metrics ──────────────────────────────────────────────────
        "Pool Size (pre-rerank)": pool_size,
        "First Match Rank (pool)": _safe(first_rank_pool),
        "Hit@1 (pool)": "Y" if hit_at_1 else "N",
        "Hit@3 (pool)": "Y" if hit_at_3 else "N",
        "Hit@5 (pool)": "Y" if hit_at_5 else "N",
        "Hit@10 (pool)": "Y" if hit_at_10 else "N",
        "Hit@Any (pool)": "Y" if hit_any else "N",
        "Pass / Fail": pass_fail,
        # ── final top-N metrics ───────────────────────────────────────────
        "Final Output Count": len(final_codes),
        "First Match Rank (final top-N)": _safe(first_rank_final),
        "Correct in Final Top-N": "Y" if final_hit else "N",
        "Top-5 Codes (final)": " / ".join(final_codes),
        # ── top-5 per-slot detail ─────────────────────────────────────────
        **top5_cols,
        # ── full pool (top-100) for reference ────────────────────────────
        "All Candidate Codes (top-100)": " / ".join(pool_codes[:100]),
    }
    return row_data


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    def _agg(sub: pd.DataFrame, label: str) -> dict[str, Any]:
        n = len(sub)
        passes = (sub["Pass / Fail"] == "PASS").sum()
        h1 = (sub["Hit@1 (pool)"] == "Y").sum()
        h3 = (sub["Hit@3 (pool)"] == "Y").sum()
        h5 = (sub["Hit@5 (pool)"] == "Y").sum()
        h10 = (sub["Hit@10 (pool)"] == "Y").sum()
        fin_hit = (sub["Correct in Final Top-N"] == "Y").sum()
        ranks = pd.to_numeric(sub["First Match Rank (pool)"], errors="coerce").dropna()
        pool_sz = pd.to_numeric(sub["Pool Size (pre-rerank)"], errors="coerce").dropna()
        return {
            "Group": label,
            "Cases": n,
            "PASS (hit@any)": passes,
            "Pass Rate %": _pct(passes, n),
            "Hit@1 count": h1,
            "Hit@1 %": _pct(h1, n),
            "Hit@3 count": h3,
            "Hit@3 %": _pct(h3, n),
            "Hit@5 count": h5,
            "Hit@5 %": _pct(h5, n),
            "Hit@10 count": h10,
            "Hit@10 %": _pct(h10, n),
            "Correct in Final Top-5": fin_hit,
            "Correct in Final Top-5 %": _pct(fin_hit, n),
            "Avg First Match Rank": round(ranks.mean(), 2) if len(ranks) else "N/A",
            "Median First Match Rank": (
                round(ranks.median(), 2) if len(ranks) else "N/A"
            ),
            "Avg Pool Size": round(pool_sz.mean(), 2) if len(pool_sz) else "N/A",
            "Median Pool Size": round(pool_sz.median(), 2) if len(pool_sz) else "N/A",
            "P90 Pool Size": round(pool_sz.quantile(0.9), 2) if len(pool_sz) else "N/A",
        }

    rows = [_agg(df, "OVERALL")]
    for cls in sorted(df[COL_CLASS].dropna().unique()):
        rows.append(_agg(df[df[COL_CLASS] == cls], cls))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def evaluate() -> None:
    if not TEST_FILE.exists():
        print(f"ERROR: test file not found: {TEST_FILE}")
        sys.exit(1)

    df_raw = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET, header=HEADER_ROW)
    df_raw = df_raw.dropna(subset=[COL_EXP, COL_QUERY]).copy()

    available = len(df_raw)
    cases_df = df_raw.head(min(TARGET_CASES, available))
    total = len(cases_df)

    print(f"\n{'='*60}")
    print(f"  LOINC Detailed Evaluation — No-Rerank Pipeline")
    print(f"{'='*60}")
    print(f"  Test cases available : {available}")
    print(f"  Evaluating           : {total}")
    print(f"  Results directory    : {RESULTS_DIR}")
    print(f"{'='*60}\n")

    result_rows: list[dict[str, Any]] = []
    passed = failed = 0

    for idx, (_, row) in enumerate(cases_df.iterrows(), start=1):
        case_id = str(row.get(COL_ID, idx))
        print(f"  [{idx:>2}/{total}] {case_id}", end="", flush=True)

        result = await _evaluate_case(row)
        if result is None:
            print("  (skipped — empty query)")
            continue

        status = result["Pass / Fail"]
        pool = result["Pool Size (pre-rerank)"]
        rank = result["First Match Rank (pool)"]
        fin = result["Correct in Final Top-N"]

        if status == "PASS":
            passed += 1
            print(f"  ✓ PASS  pool={pool:>4}  pool_rank={rank:<6} final_hit={fin}")
        else:
            failed += 1
            print(f"  ✗ FAIL  pool={pool:>4}")

        result_rows.append(result)

    # ── Write detail CSV ──────────────────────────────────────────────────
    detail_df = pd.DataFrame(result_rows)
    detail_df.to_csv(DETAIL_CSV, index=False)

    # ── Build & write summary CSV ─────────────────────────────────────────
    summary_df = _build_summary(detail_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # ── Console summary ───────────────────────────────────────────────────
    overall = summary_df.iloc[0]
    n = int(overall["Cases"])

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Evaluated cases      : {n}")
    print(
        f"  PASS (hit@any)       : {overall['PASS (hit@any)']}  ({overall['Pass Rate %']}%)"
    )
    print(f"  Hit@1  (rank 1)      : {overall['Hit@1 count']}  ({overall['Hit@1 %']}%)")
    print(f"  Hit@3  (top-3)       : {overall['Hit@3 count']}  ({overall['Hit@3 %']}%)")
    print(f"  Hit@5  (top-5)       : {overall['Hit@5 count']}  ({overall['Hit@5 %']}%)")
    print(
        f"  Hit@10 (top-10)      : {overall['Hit@10 count']}  ({overall['Hit@10 %']}%)"
    )
    print(
        f"  Correct in final top-{TOP_N_FINAL}: "
        f"{overall['Correct in Final Top-5']}  ({overall['Correct in Final Top-5 %']}%)"
    )
    print(f"  Avg first match rank : {overall['Avg First Match Rank']}")
    print(f"  Avg pool size        : {overall['Avg Pool Size']}")
    print(f"\n  By class:")
    for _, r in summary_df.iloc[1:].iterrows():
        print(
            f"    {r['Group']:<12}  cases={int(r['Cases']):>2}  pass={r['Pass Rate %']:>5}%"
            f"  hit@5={r['Hit@5 %']:>5}%  final_hit={r['Correct in Final Top-5 %']:>5}%"
        )

    print(f"\n  Detail CSV  → {DETAIL_CSV}")
    print(f"  Summary CSV → {SUMMARY_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(evaluate())
