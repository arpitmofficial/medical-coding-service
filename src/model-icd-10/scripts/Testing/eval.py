import asyncio
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Setup path to import from app module (assumes this script is in a subfolder of the main project)
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MODEL_DIR))

from app.adaptive_retrieval import adaptive_retrieve_icd_candidates
from app.execution_analysis import tracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Output to results/ subfolder inside the current subfolder
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_SET_PATH = SCRIPT_DIR / "medical_coding_test_set.csv"
RESULTS_PATH = RESULTS_DIR / "eval_results.csv"
SUMMARY_PATH = RESULTS_DIR / "eval_summary.csv"

# Rate-limit handling
MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "5"))
BASE_BACKOFF_SECONDS = float(os.getenv("EVAL_BASE_BACKOFF_SECONDS", "2.0"))
QUERY_THROTTLE_SECONDS = float(os.getenv("EVAL_QUERY_THROTTLE_SECONDS", "0.5"))


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "429",
            "rate limit",
            "rate-limited",
            "too many requests",
        )
    )


class RateLimitExhausted(RuntimeError):
    """Raised when the evaluator should stop and resume later."""


class StopEvaluation(RuntimeError):
    """Raised when the current run should stop and be resumed later."""


def _get_tracker_module_error(module_name: str) -> str | None:
    """Return the last recorded error for a tracker module, if any."""
    module = tracker._modules.get(module_name)
    if not module or not module.api_calls:
        return None
    return module.api_calls[-1].error


def _should_stop_on_llm_error(error_msg: str | None) -> bool:
    """Decide whether the preprocessing LLM error should stop the batch."""
    if not error_msg:
        return False

    lower = error_msg.lower()
    return any(
        marker in lower
        for marker in (
            "llm rate-limited / high traffic",
            "llm api timeout",
        )
    )


def parse_expected_codes(codes_str: str) -> set[str]:
    """Parse expected codes from CSV (e.g., 'I21.09' or 'I21.09 / I44.30 / R57.0')."""
    codes = set()
    for code in codes_str.split("/"):
        code = code.strip()
        if code:
            codes.add(code)
    return codes


def _get_total_tokens_from_tracker() -> int:
    """Sum up total tokens from all recorded API calls."""
    total = 0
    for module_metrics in tracker._modules.values():
        for call in module_metrics.api_calls:
            if call.total_tokens is not None:
                total += call.total_tokens
    return total


async def evaluate_single(diagnosis: str, expected_codes: set[str]) -> dict[str, Any]:
    """Run retrieval for a single diagnosis and compute metrics."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        tracker.reset()  # Clear previous tracking
        tracker.pipeline_start = time.perf_counter()

        try:
            results = await adaptive_retrieve_icd_candidates(diagnosis)
            tracker.pipeline_end = time.perf_counter()
            elapsed = tracker.pipeline_end - tracker.pipeline_start

            preprocessing_error = _get_tracker_module_error("preprocessing.py")
            if _should_stop_on_llm_error(preprocessing_error):
                raise StopEvaluation(
                    preprocessing_error or "LLM rate-limited / high traffic"
                )

            # Extract codes from results (assuming each result has 'code' field)
            returned_codes = [r.get("code") for r in results if "code" in r]

            # Metrics
            in_top_5 = any(code in expected_codes for code in returned_codes[:5])
            rank_1 = returned_codes[0] in expected_codes if returned_codes else False
            failed = not any(code in expected_codes for code in returned_codes)

            # First matching code rank (1-indexed, or None if not found)
            first_match_rank = None
            for idx, code in enumerate(returned_codes):
                if code in expected_codes:
                    first_match_rank = idx + 1
                    break

            # Get total tokens from tracker's API calls
            tokens_used = _get_total_tokens_from_tracker()

            return {
                "diagnosis": diagnosis,
                "expected_codes": " / ".join(sorted(expected_codes)),
                "returned_codes": " / ".join(returned_codes[:5]),  # Top 5 returned
                "in_top_5": "" if in_top_5 else "",
                "rank_1": "" if rank_1 else "",
                "failed": "" if failed else "",
                "first_match_rank": first_match_rank if first_match_rank else "N/A",
                "time_seconds": round(elapsed, 3),
                "tokens_used": tokens_used,
                "attempts": attempt,
            }
        except Exception as e:
            tracker.pipeline_end = time.perf_counter()
            elapsed = tracker.pipeline_end - tracker.pipeline_start
            last_error = e
            if isinstance(e, StopEvaluation):
                raise
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limited on '%s' (attempt %d/%d). Sleeping %.1fs before retry...",
                    diagnosis,
                    attempt,
                    MAX_RETRIES,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue

            logger.error(f"Retrieval failed for '{diagnosis}': {e}")
            return {
                "diagnosis": diagnosis,
                "expected_codes": " / ".join(sorted(expected_codes)),
                "returned_codes": "ERROR",
                "in_top_5": "",
                "rank_1": "",
                "failed": "",
                "first_match_rank": "ERROR",
                "time_seconds": round(elapsed, 3),
                "tokens_used": 0,
                "attempts": attempt,
                "error": str(e),
            }

    # Should only happen if retries were exhausted on rate limits.
    # Raise a stop signal so the caller can persist previous results and resume later.
    raise RateLimitExhausted(
        str(last_error) if last_error else "rate limit retries exhausted"
    )


def _save_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Persist results and return the dataframe used for summary output."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    return results_df


async def run_eval():
    """Run evaluation on all test cases."""
    # Load existing results for resume capability
    existing_results = []
    tested_diagnoses = set()
    if RESULTS_PATH.exists():
        try:
            existing_results_df = pd.read_csv(RESULTS_PATH)
            existing_results = existing_results_df.to_dict("records")
            tested_diagnoses = set(existing_results_df["diagnosis"].str.strip())
            logger.info(
                " Resuming: found %d existing results. Skipping %d tested diagnoses.",
                len(existing_results),
                len(tested_diagnoses),
            )
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}. Starting fresh.")
            tested_diagnoses = set()

    if not TEST_SET_PATH.exists():
        logger.error(f"Test set not found: {TEST_SET_PATH}")
        return

    logger.info(f"Loading test set from {TEST_SET_PATH}")
    df = pd.read_csv(TEST_SET_PATH)

    # Column names might vary; try common patterns
    diagnosis_col = next(
        (c for c in df.columns if "diagnosis" in c.lower()), df.columns[0]
    )
    codes_col = next((c for c in df.columns if "code" in c.lower()), df.columns[1])

    logger.info(
        f"Using diagnosis column: '{diagnosis_col}', codes column: '{codes_col}'"
    )

    results = existing_results.copy()
    total = len(df)
    skipped_count = 0
    stopped_early = False

    for idx, row in df.iterrows():
        diagnosis = str(row[diagnosis_col]).strip()
        expected_codes_str = str(row[codes_col]).strip()
        expected_codes = parse_expected_codes(expected_codes_str)

        if not diagnosis or not expected_codes:
            logger.warning(f"Skipping row {idx}: empty diagnosis or codes")
            skipped_count += 1
            continue

        if diagnosis in tested_diagnoses:
            logger.debug(f"Skipping row {idx}: already tested")
            skipped_count += 1
            continue

        logger.info(f"[{idx+1}/{total}] Testing: {diagnosis[:80]}")

        try:
            result = await evaluate_single(diagnosis, expected_codes)
        except StopEvaluation as e:
            stopped_early = True
            logger.warning(
                "Stopping evaluation on '%s' due to LLM rate-limit/high-traffic fallback: %s",
                diagnosis,
                e,
            )
            break
        except RateLimitExhausted as e:
            stopped_early = True
            logger.warning(
                "Rate limit exhausted while testing '%s'. Saving previous results and stopping here: %s",
                diagnosis,
                e,
            )
            break

        results.append(result)
        _save_results(results)
        logger.info(f" Progress saved to {RESULTS_PATH} ({len(results)} total rows)")

        if QUERY_THROTTLE_SECONDS > 0:
            await asyncio.sleep(QUERY_THROTTLE_SECONDS)

    # Final save (covers the case where we stopped early after some writes)
    if results:
        results_df = _save_results(results)
        logger.info(
            " Results saved to %s (%d total, %d new)",
            RESULTS_PATH,
            len(results),
            len(results) - len(existing_results),
        )

        # Print summary stats
        print("\n" + "=" * 100)
        print("EVALUATION SUMMARY")
        print("=" * 100)
        print(results_df.to_string(index=False))
        print("=" * 100)

        # Summary metrics
        in_top_5_count = (results_df["in_top_5"] == "").sum()
        rank_1_count = (results_df["rank_1"] == "").sum()
        failed_count = (results_df["failed"] == "").sum()
        total_count = len(results)

        in_top_5_pct = (100 * in_top_5_count / total_count) if total_count else 0.0
        rank_1_pct = (100 * rank_1_count / total_count) if total_count else 0.0
        failed_pct = (100 * failed_count / total_count) if total_count else 0.0

        avg_time = results_df["time_seconds"].mean()
        median_time = results_df["time_seconds"].median()
        avg_tokens = results_df["tokens_used"].mean()
        median_tokens = results_df["tokens_used"].median()

        summary_row = {
            "total_queries": total_count,
            "top5_hits": int(in_top_5_count),
            "top5_pct": round(in_top_5_pct, 2),
            "rank1_hits": int(rank_1_count),
            "rank1_pct": round(rank_1_pct, 2),
            "failed_cases": int(failed_count),
            "failed_pct": round(failed_pct, 2),
            "avg_time_seconds": round(float(avg_time), 3),
            "median_time_seconds": round(float(median_time), 3),
            "avg_tokens_used": round(float(avg_tokens), 2),
            "median_tokens_used": round(float(median_tokens), 2),
        }
        pd.DataFrame([summary_row]).to_csv(SUMMARY_PATH, index=False)
        logger.info(f" Summary metrics saved to {SUMMARY_PATH}")

        print(f"\nMetrics:")
        print(f"  In Top 5:     {in_top_5_count}/{total_count} ({in_top_5_pct:.1f}%)")
        print(f"  Rank 1:       {rank_1_count}/{total_count} ({rank_1_pct:.1f}%)")
        print(f"  Failed:       {failed_count}/{total_count} ({failed_pct:.1f}%)")
        print(f"  Skipped:      {skipped_count} (empty or already tested)")
        print(f"  New results:  {len(results)}")
        print(f"  Avg Time:     {avg_time:.3f}s")
        print(f"  Median Time:  {median_time:.3f}s")
        print(f"  Avg Tokens:   {avg_tokens:.2f}")
        print(f"  Median Tokens:{median_tokens:.2f}")
        if stopped_early:
            print(
                "  NOTE: Stopped early due to rate limiting; rerun to resume from the next untested row."
            )
        print("=" * 100 + "\n")
    else:
        logger.error("No valid test cases to evaluate.")


if __name__ == "__main__":
    asyncio.run(run_eval())
