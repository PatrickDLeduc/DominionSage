"""
orchestrator.py — DominionSage Retrieval Orchestrator (Phase 3)

This is the central coordinator of the entire RAG pipeline. It:
  1. Receives a user question + optional conversation history
  2. Rewrites the query to resolve pronouns/references (NEW)
  3. Routes it to the correct retrieval path (via the router)
  4. Fetches context from the card DB, vector store, or both
  5. Passes the context to the synthesizer for final answer generation
  6. Returns the answer + sources for the UI to display

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
from retrieval.rewriter import rewrite_query


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def answer_question(
    query: str,
    expansion: str | None = None,
    conversation_history: list[dict] | None = None,
    kingdom_context: str | None = None,
) -> dict:
    """
    Full pipeline: rewrite → route → retrieve → synthesize.

    Args:
        query:                The user's natural language question.
        expansion:            Optional expansion filter (e.g., "Seaside").
        conversation_history: Optional list of recent messages, each with
                              "role" ("user"/"assistant") and "content".
                              Used to resolve follow-up references like
                              "it", "that card", "which ones", etc.
        kingdom_context:      Optional string describing the active kingdom
                              cards (from Kingdom Advisor). Injected into
                              the synthesizer prompt so the chat can answer
                              questions about the current game setup.

    Returns:
        Dict with:
          - answer:         The generated answer string
          - sources:        List of source dicts (for the UI's source panel)
          - query_type:     Which route was taken (for debugging/evals)
          - original_query: The user's original query (before rewriting)
          - rewritten_query: The resolved query (after rewriting, if changed)
    """
    # Step 0: Rewrite the query to resolve any follow-up references
    original_query = query
    query = rewrite_query(query, conversation_history)

    # Step 1: Classify the (possibly rewritten) query
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
    answer = synthesize_answer(query, context, kingdom_context=kingdom_context)

    # Step 4: Append any meta notes directly to the answer
    for source in context["sources"]:
        if source["type"] == "meta":
            answer += f"\n\n> **Note:** {source['data']['note']}"

    result = {
        "answer": answer,
        "sources": context["sources"],
        "query_type": query_type,
    }

    # Include rewriting info for debugging and the UI
    if query != original_query:
        result["original_query"] = original_query
        result["rewritten_query"] = query

    return result


# ─────────────────────────────────────────────────────────────────
# Retrieval strategies (unchanged)
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

    if expansion and "expansion" not in filters:
        filters["expansion"] = expansion

    if not filters:
        filters = {}

    cards = filtered_search(filters)

    total_found = len(cards)
    if total_found > 20:
        cards = cards[:20]

    sources = [{"type": "card_db", "data": c} for c in cards]

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
    Queries BOTH the card database and the vector store.
    """
    sources = []

    card_name = find_card_name_in_query(query)
    if card_name:
        cards = lookup_card(card_name)
        sources.extend([{"type": "card_db", "data": c} for c in cards])

    chunks = search_rules(query, top_k=3, expansion=expansion)
    sources.extend([{"type": "rulebook", "data": c} for c in chunks])

    return sources


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test without history (should work exactly as before)
    print("=" * 60)
    print("Test 1: No conversation history (baseline)")
    print("=" * 60)

    result = answer_question("What does Chapel do?")
    print(f"Type: {result['query_type']}")
    print(f"Answer: {result['answer'][:200]}...")

    # Test with history (follow-up question)
    print(f"\n{'=' * 60}")
    print("Test 2: Follow-up with pronoun resolution")
    print("=" * 60)

    history = [
        {"role": "user", "content": "What does Throne Room do?"},
        {"role": "assistant", "content": "Throne Room costs 4 and lets you play an Action card from your hand twice."},
    ]

    result = answer_question("What combos well with it?", conversation_history=history)
    print(f"Type: {result['query_type']}")
    if "rewritten_query" in result:
        print(f"Original:  {result['original_query']}")
        print(f"Rewritten: {result['rewritten_query']}")
    print(f"Answer: {result['answer'][:200]}...")
