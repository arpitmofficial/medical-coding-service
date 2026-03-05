import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Enable logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

from app.retrieval import retrieve_icd_candidates


async def main() -> None:
    while True:
        query = input("\nEnter clinical note (or 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        print("\nRunning two-stage pipeline …\n")
        results = await retrieve_icd_candidates(query)

        if not results:
            print("No results found above the minimum score threshold.")
            continue

        print(f"Top {len(results)} ICD-10 code(s):\n")
        for i, r in enumerate(results, start=1):
            print(
                f"  {i}. [{r['code']}] {r['description']}\n"
                f"     Confidence : {r.get('confidence', 'N/A')}%\n"
                f"     Explanation: {r.get('explanation', '')}\n"
            )


if __name__ == "__main__":
    asyncio.run(main())