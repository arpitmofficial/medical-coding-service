import asyncio
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import logging and tracking systems
from app.config import console_logger
from app.execution_analysis import tracker
from app.adaptive_retrieval import adaptive_retrieve_icd_candidates

print("ICD-10 Medical Coding Service")
print("=" * 40)


async def main() -> None:
    while True:
        query = input("\nEnter clinical note (or 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        print()  # Clean spacing
        
        tracker.reset()
        tracker.pipeline_start = time.perf_counter()
        try:
            results = await adaptive_retrieve_icd_candidates(query)
        except Exception as exc:
            tracker.pipeline_end = time.perf_counter()
            err_type = type(exc).__name__
            print(f"\n  [ERROR] Pipeline failed — {err_type}: {exc}")
            tracker.print_report()
            continue
        tracker.pipeline_end = time.perf_counter()

        if not results:
            print("No results found above the minimum confidence threshold.")
            tracker.print_report()
            continue

        print(f"\nTop {len(results)} ICD-10 code(s):")
        print("-" * 60)
        for i, r in enumerate(results, start=1):
            print(
                f"  {i}. [{r['code']}] {r['description']}\n"
                f"     Confidence : {r.get('confidence', 'N/A')}%\n"
                f"     Explanation: {r.get('explanation', '')}\n"
            )

        tracker.print_report()


if __name__ == "__main__":
    asyncio.run(main())