import asyncio
import sys
import time
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_LOINC_DIR = SCRIPT_DIR.parent
MODEL_ICD10_DIR = MODEL_LOINC_DIR.parent / "model-icd-10"

# Add LOINC app path for LOINC-specific modules
sys.path.insert(0, str(MODEL_LOINC_DIR / "app"))

# Import LOINC retrieval module
from adaptive_retrieval_loinc import adaptive_retrieve_loinc_candidates

print("LOINC Lab/Observation Coding Service")
print("=" * 40)


async def main() -> None:
    while True:
        query = input("\nEnter lab/observation query (or 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        print()
        t0 = time.perf_counter()
        try:
            results = await adaptive_retrieve_loinc_candidates(query)
        except Exception as exc:
            err_type = type(exc).__name__
            print(f"\n  [ERROR] Pipeline failed - {err_type}: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        if not results:
            print("No results found above the minimum confidence threshold.")
            continue

        print(f"\nTop {len(results)} LOINC code(s):")
        print("-" * 60)
        for i, r in enumerate(results, start=1):
            print(
                f"  {i}. [{r['code']}] {r['description']}\n"
                f"     Confidence : {r.get('confidence', 'N/A')}%\n"
                f"     Explanation: {r.get('explanation', '')}\n"
            )

        print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
