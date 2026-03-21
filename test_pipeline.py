"""
test_pipeline.py — Quick test of the full DominionSage pipeline.

Run from the project root:
  python test_pipeline.py

Tests all 4 query types and prints the results.
"""

from retrieval.orchestrator import answer_question


def main():
    test_questions = [
        # Type 1: Card lookup
        "What does Chapel do?",

        # Type 2: Filtered search
        "Show me all Action cards costing 4 or less",

        # Type 3: Rules question
        "When can I play Reaction cards?",

        # Type 4: Strategy / combo
        "What combos well with Throne Room?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print(f"{'=' * 60}")

        result = answer_question(q)

        print(f"Route: {result['query_type']}")
        print(f"Sources ({len(result['sources'])}):")
        for s in result["sources"]:
            if s["type"] == "card_db":
                card = s["data"]
                print(f"  🃏 {card['name']} (${card['cost']}, {card['type']})")
            elif s["type"] == "rulebook":
                chunk = s["data"]
                sim = chunk.get("similarity", 0)
                print(f"  📄 {chunk.get('expansion', '?')} p.{chunk.get('source_page', '?')} (sim: {sim:.3f})")

        print(f"\nAnswer:\n{result['answer']}")

    print(f"\n{'=' * 60}")
    print("All 4 query types tested. If the answers look good, Phase 3 is complete!")
    print("Next step: python app/main.py (Phase 4 — Streamlit UI)")


if __name__ == "__main__":
    main()
