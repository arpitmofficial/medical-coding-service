import asyncio
import sys
import time
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CPT_DIR = SCRIPT_DIR.parent
MODEL_ICD10_DIR = MODEL_CPT_DIR.parent / "model-icd-10"

# Add ICD-10 path for shared modules
sys.path.insert(0, str(MODEL_ICD10_DIR))

# Import shared modules from ICD-10
from app.config import console_logger
from app.execution_analysis import tracker

# Add CPT app path for CPT-specific modules
sys.path.insert(0, str(MODEL_CPT_DIR / "app"))

# Import CPT retrieval module
from adaptive_retrieval_cpt import adaptive_retrieve_cpt_candidates

print("CPT Medical Procedure Coding Service")
print("=" * 40)


async def main() -> None:
    while True:
        query = input("\nEnter procedure description (or 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        print()  # Clean spacing
        
        tracker.reset()
        tracker.pipeline_start = time.perf_counter()
        try:
            results = await adaptive_retrieve_cpt_candidates(query)
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

        print(f"\nTop {len(results)} CPT code(s):")
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
