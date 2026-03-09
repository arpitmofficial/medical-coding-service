import asyncio
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Enable logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)

from app.execution_analysis import tracker
from app.retrieval import retrieve_icd_candidates


async def main() -> None:
    while True:
        query = input("\nEnter clinical note (or 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        print("\nRunning two-stage pipeline …\n")

        tracker.reset()
        tracker.pipeline_start = time.perf_counter()
        try:
            results = await retrieve_icd_candidates(query)
        except Exception as exc:
            tracker.pipeline_end = time.perf_counter()
            err_type = type(exc).__name__
            print(f"\n  [ERROR] Pipeline failed — {err_type}: {exc}")
            print(  "  Reason: An unrecoverable error occurred in one of the API calls.")
            print(  "  The error has been recorded in the execution report below.\n")
            tracker.print_report()
            continue
        tracker.pipeline_end = time.perf_counter()

        if not results:
            print("No results found above the minimum score threshold.")
            tracker.print_report()
            continue

        print(f"Top {len(results)} ICD-10 code(s):\n")
        for i, r in enumerate(results, start=1):
            print(
                f"  {i}. [{r['code']}] {r['description']}\n"
                f"     Confidence : {r.get('confidence', 'N/A')}%\n"
                f"     Explanation: {r.get('explanation', '')}\n"
            )

        tracker.print_report()


if __name__ == "__main__":
    asyncio.run(main())