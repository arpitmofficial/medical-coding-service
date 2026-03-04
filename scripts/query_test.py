import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.retrieval import retrieve_icd_candidates

while True:

    query = input("\nEnter medical query: ")

    results = retrieve_icd_candidates(query)

    print("\nTop Matches:\n")

    for r in results[:10]:

        print(f"{r['code']} - {r['description']}  (score={r['score']:.3f})")