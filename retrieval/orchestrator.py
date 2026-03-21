"""
orchestrator.py — DominionSage Retrieval Orchestrator (Phase 3)

This is the central coordinator of the entire RAG pipeline. It:
  1. Receives a user question
  2. Routes it to the correct retrieval path (via the router)
  3. Fetches context from the card DB, vector store, or both
  4. Passes the context to the synthesizer for final answer generation
  5. Returns the answer + sources for the UI to display

Analogy: Think of the orchestrator like a restaurant's expediter — the
person who reads incoming orders, tells each station (grill, sauté,
pastry) what to prepare, collects the finished dishes, and plates
everything together before it goes to the table.

The orchestrator doesn't cook anything. It coordinates.
"""

from retrieval.router import classify_query, find_card_name_in_query, parse_filters
from retrieval.card_lookup import lookup_card, filtered_search
from retrieval.rules_search import search_rules
from retrieval.synthesizer import synthesize_answer


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def answer_question(query: str, expansion: str | None = None) -> dict:
    """
    Full pipeline: route → retrieve → synthesize.

    Args:
        query:     The user's natural language question.
        expansion: Optional expansion filter (e.g., "Seaside").

    Returns:
        Dict with:
          - answer:     The generated answer string
          - sources:    List of source dicts (for the UI's source panel)
          - query_type: Which route was taken (for debugging/evals)
    """
    # Step 1: Classify the query
    query_type = classify_query(query)

    # Step 2: Retrieve context based on query type
    context = {
        "query_type": query_type,
        "sources": [],
    }

    if query_type == "card_lookup":
        context["sources"] = _retrieve_card_lookup(query)

    elif query_type == "filtered_search":
        context["sources"] = _retrieve_filtered_search(query, expansion)

    elif query_type == "rules_question":
        context["sources"] = _retrieve_rules(query, expansion)

    elif query_type == "strategy_combo":
        context["sources"] = _retrieve_strategy_combo(query, expansion)

    # Step 3: Synthesize the final answer
    answer = synthesize_answer(query, context)

    # Step 4: Append any meta notes directly to the answer
    # (Don't rely on the LLM to surface these — it sometimes ignores them)
    for source in context["sources"]:
        if source["type"] == "meta":
            answer += f"\n\n> **Note:** {source['data']['note']}"

    return {
        "answer": answer,
        "sources": context["sources"],
        "query_type": query_type,
    }


# ─────────────────────────────────────────────────────────────────
# Retrieval strategies
# ─────────────────────────────────────────────────────────────────

def _retrieve_card_lookup(query: str) -> list[dict]:
    """
    Type 1: Direct card lookup.
    Find the specific card mentioned in the query.
    """
    card_name = find_card_name_in_query(query)

    if card_name:
        cards = lookup_card(card_name)
    else:
        # Fallback: try using the whole query as a search term
        # (handles cases like "Chapel?" where the router detected
        # a card name but find_card_name might have edge cases)
        cards = lookup_card(query.strip("?!. "))

    return [{"type": "card_db", "data": c} for c in cards]


def _retrieve_filtered_search(
    query: str,
    expansion: str | None = None,
) -> list[dict]:
    """
    Type 2: Filtered card search.
    Parse filters from the query and run a SQL search.
    """
    filters = parse_filters(query)

    # Apply the UI's expansion filter if set
    if expansion and "expansion" not in filters:
        filters["expansion"] = expansion

    # If no filters were parsed, fall back to a broad search
    # (the synthesizer will explain the results)
    if not filters:
        filters = {}

    cards = filtered_search(filters)

    # Cap results to avoid overwhelming the LLM context
    # (more than 20 cards is too much for a single answer)
    total_found = len(cards)
    if total_found > 20:
        cards = cards[:20]

    sources = [{"type": "card_db", "data": c} for c in cards]

    # If we capped results, add a metadata source so the synthesizer
    # can tell the user there are more
    if total_found > 20:
        sources.append({
            "type": "meta",
            "data": {
                "note": f"Showing 20 of {total_found} matching cards. "
                        f"Try narrowing your search with a cost range, "
                        f"expansion, or card type to see more specific results."
            },
        })

    return sources


def _retrieve_rules(
    query: str,
    expansion: str | None = None,
) -> list[dict]:
    """
    Type 3: Rules question.
    Semantic search over rulebook chunks.
    """
    chunks = search_rules(query, top_k=5, expansion=expansion)
    return [{"type": "rulebook", "data": c} for c in chunks]


def _retrieve_strategy_combo(
    query: str,
    expansion: str | None = None,
) -> list[dict]:
    """
    Type 4: Strategy / combo question.
    Queries BOTH the card database and the vector store, then
    combines the results.

    This gives the synthesizer both structured card data (exact stats)
    and rulebook knowledge (timing rules, interactions) to draw from.
    """
    sources = []

    # Card DB: find any cards mentioned in the query
    card_name = find_card_name_in_query(query)
    if card_name:
        cards = lookup_card(card_name)
        sources.extend([{"type": "card_db", "data": c} for c in cards])

    # Vector search: find relevant rulebook context
    chunks = search_rules(query, top_k=3, expansion=expansion)
    sources.extend([{"type": "rulebook", "data": c} for c in chunks])

    return sources


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "What does Chapel do?",                          # Type 1
        "Show me all Action cards costing 4 or less",    # Type 2
        "When can I play Reaction cards?",               # Type 3
        "What combos well with Throne Room?",            # Type 4
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print(f"{'=' * 60}")

        result = answer_question(q)
        print(f"Type: {result['query_type']}")
        print(f"Sources: {len(result['sources'])}")
        for s in result["sources"]:
            if s["type"] == "card_db":
                print(f"  📋 Card: {s['data']['name']}")
            elif s["type"] == "rulebook":
                print(f"  📄 Rulebook: {s['data'].get('expansion', '?')} p.{s['data'].get('source_page', '?')}")
        print(f"\nAnswer:\n{result['answer']}")
