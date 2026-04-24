import argparse
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
SWEEP_RESULTS_FILE = RESULTS_DIR / "threshold_sweep_results.csv"
SWEEP_BEST_FILE = RESULTS_DIR / "threshold_sweep_best.csv"

TEST_SHEET = "LOINC Test Cases"
HEADER_ROW = 1
DEFAULT_TARGET_CASES = 67

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _parse_expected_codes(raw: Any) -> list[str]:
    return [code.strip() for code in str(raw).split("/") if code and code.strip()]


def _load_cases(target_cases: int) -> pd.DataFrame:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    df = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET, header=HEADER_ROW)

    expected_code_col = "Expected LOINC Code"
    query_col = "Clinical Scenario / Order Description"
    id_col = "Test Case ID"
    class_col = "Class / Domain"

    required = [id_col, class_col, expected_code_col, query_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = df.dropna(subset=[expected_code_col, query_col]).copy()
    available = len(df)

    if available < target_cases:
        logger.warning(
            "Requested %d cases but only %d valid cases available; sweeping all available rows.",
            target_cases,
            available,
        )
        return df

    return df.head(target_cases)


def _build_config_grid(mode: str) -> list[dict[str, Any]]:
    # Conservative defaults + progressively stricter thresholds.
    base = [
        {
            "config_name": "baseline_dual_relaxed",
            "dense_threshold": 0.0,
            "sparse_threshold": 0.0,
            "rrf_threshold": 0.0,
            "min_score": 0.0,
            "fusion_alpha": 0.55,
            "fusion_beta": 0.35,
            "fusion_gamma": 0.10,
            "metadata_weight": 0.50,
            "qdrant_top_k": 50,
            "use_dual_threshold": True,
        },
        {
            "config_name": "balanced_low_threshold",
            "dense_threshold": 0.10,
            "sparse_threshold": 0.50,
            "rrf_threshold": 0.0,
            "min_score": 0.10,
            "fusion_alpha": 0.55,
            "fusion_beta": 0.35,
            "fusion_gamma": 0.10,
            "metadata_weight": 0.50,
            "qdrant_top_k": 50,
            "use_dual_threshold": True,
        },
        {
            "config_name": "balanced_mid_threshold",
            "dense_threshold": 0.20,
            "sparse_threshold": 1.00,
            "rrf_threshold": 0.0,
            "min_score": 0.20,
            "fusion_alpha": 0.55,
            "fusion_beta": 0.35,
            "fusion_gamma": 0.10,
            "metadata_weight": 0.50,
            "qdrant_top_k": 50,
            "use_dual_threshold": True,
        },
        {
            "config_name": "dense_lean",
            "dense_threshold": 0.20,
            "sparse_threshold": 0.50,
            "rrf_threshold": 0.0,
            "min_score": 0.20,
            "fusion_alpha": 0.65,
            "fusion_beta": 0.25,
            "fusion_gamma": 0.10,
            "metadata_weight": 0.45,
            "qdrant_top_k": 50,
            "use_dual_threshold": True,
        },
        {
            "config_name": "sparse_lean",
            "dense_threshold": 0.10,
            "sparse_threshold": 1.50,
            "rrf_threshold": 0.0,
            "min_score": 0.20,
            "fusion_alpha": 0.45,
            "fusion_beta": 0.45,
            "fusion_gamma": 0.10,
            "metadata_weight": 0.50,
            "qdrant_top_k": 50,
            "use_dual_threshold": True,
        },
        {
            "config_name": "strict_pool_control",
            "dense_threshold": 0.25,
            "sparse_threshold": 2.00,
            "rrf_threshold": 0.0,
            "min_score": 0.30,
            "fusion_alpha": 0.60,
            "fusion_beta": 0.30,
            "fusion_gamma": 0.10,
            "metadata_weight": 0.40,
            "qdrant_top_k": 50,
            "use_dual_threshold": True,
        },
    ]

    if mode == "quick":
        return base

    extended = [
        {
            "config_name": "high_recall_large_pool",
            "dense_threshold": 0.05,
            "sparse_threshold": 0.25,
            "rrf_threshold": 0.0,
            "min_score": 0.05,
            "fusion_alpha": 0.50,
            "fusion_beta": 0.40,
            "fusion_gamma": 0.10,
            "metadata_weight": 0.55,
            "qdrant_top_k": 60,
            "use_dual_threshold": True,
        },
        {
            "config_name": "rrf_guardrail",
            "dense_threshold": 0.10,
            "sparse_threshold": 1.00,
            "rrf_threshold": 0.01,
            "min_score": 0.20,
            "fusion_alpha": 0.50,
            "fusion_beta": 0.30,
            "fusion_gamma": 0.20,
            "metadata_weight": 0.50,
            "qdrant_top_k": 50,
            "use_dual_threshold": True,
        },
        {
            "config_name": "weighted_only_no_dual_gate",
            "dense_threshold": 0.0,
            "sparse_threshold": 0.0,
            "rrf_threshold": 0.0,
            "min_score": 0.25,
            "fusion_alpha": 0.55,
            "fusion_beta": 0.35,
            "fusion_gamma": 0.10,
            "metadata_weight": 0.50,
            "qdrant_top_k": 50,
            "use_dual_threshold": False,
        },
        {
            "config_name": "strict_compact_pool",
            "dense_threshold": 0.30,
            "sparse_threshold": 2.50,
            "rrf_threshold": 0.01,
            "min_score": 0.35,
            "fusion_alpha": 0.65,
            "fusion_beta": 0.20,
            "fusion_gamma": 0.15,
            "metadata_weight": 0.35,
            "qdrant_top_k": 40,
            "use_dual_threshold": True,
        },
    ]

    return base + extended


async def _evaluate_config(
    cases_df: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    expected_code_col = "Expected LOINC Code"
    query_col = "Clinical Scenario / Order Description"

    async def _evaluate_row(row: pd.Series) -> tuple[bool, int, int | None] | None:
        expected_codes = _parse_expected_codes(row[expected_code_col])
        query = str(row[query_col]).strip()
        if not query:
            return None

        try:
            candidates = await retrieve_loinc_candidates_before_rerank(
                query,
                qdrant_top_k=int(config["qdrant_top_k"]),
                min_score=float(config["min_score"]),
                dense_threshold=float(config["dense_threshold"]),
                sparse_threshold=float(config["sparse_threshold"]),
                rrf_threshold=float(config["rrf_threshold"]),
                use_dual_threshold=bool(config["use_dual_threshold"]),
                fusion_alpha=float(config["fusion_alpha"]),
                fusion_beta=float(config["fusion_beta"]),
                fusion_gamma=float(config["fusion_gamma"]),
                metadata_weight=float(config["metadata_weight"]),
            )
        except Exception as exc:
            logger.warning(
                "Config %s failed on query '%s': %s",
                config["config_name"],
                query[:60],
                exc,
            )
            candidates = []

        output_codes = [
            str(c.get("code", "")).strip() for c in candidates if c.get("code")
        ]
        pool_size = len(output_codes)
        first_match_rank = next(
            (
                idx + 1
                for idx, code in enumerate(output_codes)
                if code in expected_codes
            ),
            None,
        )
        return first_match_rank is not None, pool_size, first_match_rank

    semaphore = asyncio.Semaphore(12)

    async def _bounded_evaluate_row(
        row: pd.Series,
    ) -> tuple[bool, int, int | None] | None:
        async with semaphore:
            return await _evaluate_row(row)

    tasks = [_bounded_evaluate_row(row) for _, row in cases_df.iterrows()]
    results = [result for result in await asyncio.gather(*tasks) if result is not None]

    pass_count = sum(1 for passed, _, _ in results if passed)
    total = len(results)
    pool_sizes = [pool_size for _, pool_size, _ in results]
    ranks = [rank for passed, _, rank in results if passed and rank is not None]

    pass_pct = (100.0 * pass_count / total) if total else 0.0
    avg_pool = (sum(pool_sizes) / len(pool_sizes)) if pool_sizes else 0.0
    median_pool = float(pd.Series(pool_sizes).median()) if pool_sizes else 0.0
    p90_pool = float(pd.Series(pool_sizes).quantile(0.9)) if pool_sizes else 0.0
    avg_rank = (sum(ranks) / len(ranks)) if ranks else None

    return {
        **config,
        "evaluated_cases": total,
        "pass_count": pass_count,
        "pass_pct": round(pass_pct, 2),
        "avg_pool_size": round(avg_pool, 2),
        "median_pool_size": round(median_pool, 2),
        "p90_pool_size": round(p90_pool, 2),
        "avg_first_match_rank": round(avg_rank, 2) if avg_rank is not None else "N/A",
    }


async def main(mode: str, target_cases: int) -> None:
    cases_df = _load_cases(target_cases)
    configs = _build_config_grid(mode)

    logger.info(
        "Running threshold sweep | mode=%s | configs=%d | cases=%d",
        mode,
        len(configs),
        len(cases_df),
    )

    rows: list[dict[str, Any]] = []
    for idx, config in enumerate(configs, start=1):
        logger.info("[%d/%d] Evaluating %s", idx, len(configs), config["config_name"])
        metrics = await _evaluate_config(cases_df, config)
        rows.append(metrics)

        progress_df = pd.DataFrame(rows)
        progress_df = progress_df.sort_values(
            by=["pass_pct", "avg_pool_size", "avg_first_match_rank"],
            ascending=[False, True, True],
            na_position="last",
        )
        progress_df.to_csv(SWEEP_RESULTS_FILE, index=False)
        progress_df.head(1).to_csv(SWEEP_BEST_FILE, index=False)
        logger.info(
            "Progress saved after %s | current best=%s | pass_pct=%s | avg_pool=%s",
            config["config_name"],
            progress_df.iloc[0]["config_name"],
            progress_df.iloc[0]["pass_pct"],
            progress_df.iloc[0]["avg_pool_size"],
        )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        by=["pass_pct", "avg_pool_size", "avg_first_match_rank"],
        ascending=[False, True, True],
        na_position="last",
    )
    result_df.to_csv(SWEEP_RESULTS_FILE, index=False)

    best_df = result_df.head(1)
    best_df.to_csv(SWEEP_BEST_FILE, index=False)

    logger.info("Sweep results saved to %s", SWEEP_RESULTS_FILE)
    logger.info("Best config saved to %s", SWEEP_BEST_FILE)

    print("\nTop 5 configs:")
    print(result_df.head(5).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep LOINC pre-rerank threshold/fusion settings."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick: fewer configs (faster), full: broader config grid",
    )
    parser.add_argument(
        "--target-cases",
        type=int,
        default=DEFAULT_TARGET_CASES,
        help="How many test rows to evaluate from the top of the test sheet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(mode=args.mode, target_cases=args.target_cases))
