"""
router.py — DominionSage Query Router (Phase 3)

Classifies incoming questions into one of four query types, which determines
which retrieval path the orchestrator uses.

Architecture decision: Rules-based over LLM-based.
  - Adds ~0ms latency and $0 cost per query
  - Predictable and debuggable (you can see exactly which rule fired)
  - Sufficient for 4 well-defined query types
  - You can always upgrade to an LLM-based router later if evals show
    misclassification issues

Analogy: This is like a database query planner deciding between an index
scan and a sequential scan. The planner doesn't need to be an AI — it
just needs to recognize patterns in the query and pick the right strategy.

Query Types:
  1. card_lookup     — "What does Chapel do?" → SQL exact match
  2. filtered_search — "Show me all cards costing 4" → SQL with WHERE
  3. rules_question  — "When can I play Reactions?" → vector search
  4. strategy_combo  — "What combos with Throne Room?" → both paths
"""

import os
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# Card name cache
# ─────────────────────────────────────────────────────────────────

# Load known card names so the router can detect when a specific card
# is mentioned. This avoids needing an LLM just to extract card names.

_card_names: list[str] = []


def _load_card_names() -> list[str]:
    """Load card names from the scraped data or Supabase."""
    global _card_names
    if _card_names:
        return _card_names

    # Try loading from the local JSON first (fastest, no DB call)
    cards_json = Path("data/cards.json")
    if cards_json.exists():
        with open(cards_json, "r", encoding="utf-8") as f:
            cards = json.load(f)
        _card_names = [c["name"].lower() for c in cards]
    else:
        # Fallback: hardcoded list of common cards
        # (you'd replace this with a Supabase query in production)
        _card_names = [
            "cellar", "chapel", "moat", "harbinger", "merchant", "vassal",
            "village", "workshop", "bureaucrat", "gardens", "militia",
            "moneylender", "poacher", "remodel", "smithy", "throne room",
            "bandit", "council room", "festival", "laboratory", "library",
            "market", "mine", "sentry", "witch", "artisan",
            "copper", "silver", "gold", "estate", "duchy", "province", "curse",
            "haven", "lighthouse", "native village", "astrolabe",
            "fishing village", "lookout", "monkey", "sea chart", "smugglers",
            "warehouse", "blockade", "caravan", "cutpurse", "island",
            "sailor", "salvager", "tide pools", "treasure map", "bazaar",
            "corsair", "merchant ship", "pirate", "sea witch", "tactician",
            "treasury", "wharf",
        ]

    return _card_names


def find_card_name_in_query(query: str) -> str | None:
    """
    Check if the query mentions a known card name.
    Returns the card name if found, None otherwise.

    Checks longer names first so "Throne Room" matches before "Room".
    """
    card_names = _load_card_names()
    query_lower = query.lower()

    # Sort by length descending so multi-word names match first
    for name in sorted(card_names, key=len, reverse=True):
        if name in query_lower:
            return name

    return None


# ─────────────────────────────────────────────────────────────────
# Query classification
# ─────────────────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Classify a query into one of four types.

    The order of checks matters — more specific patterns are checked
    first, with the broadest category (strategy_combo) as the default.
    """
    query_lower = query.lower()
    has_card = find_card_name_in_query(query) is not None

    # ── Type 3: Rules question ──
    # CHECK THIS FIRST — rules questions often mention card types
    # ("Reaction cards", "Duration cards") but are asking about RULES,
    # not requesting a list. The key signals are question words about
    # mechanics and timing.
    rules_signals = [
        "rule", "when can", "when do", "am i allowed",
        "can i", "can you", "is it legal", "what happens if",
        "what happens when", "phase", "turn order", "turn structure",
        "action phase", "buy phase", "cleanup", "clean up",
        "setup", "set up", "how to play", "how do you play",
        "how many players", "end of game", "game end",
        "supply pile", "empty pile", "gain a card",
        "in response", "timing", "resolve",
        "how do .* work", "how does .* work",
        "what is the turn", "what are the rules",
        "when .* play", "when .* played",
        "piles are empty", "piles empty",
    ]
    if any(signal in query_lower for signal in rules_signals):
        return "rules_question"

    # Also catch "How do [Type] cards work?" patterns specifically
    import re as _re
    if _re.search(r"how (do|does) .+ cards? work", query_lower):
        return "rules_question"

    # "When can I play [Type] cards?" is always a rules question
    if _re.search(r"when can .+ play .+ cards?", query_lower):
        return "rules_question"

    # ── Type 2: Filtered search ──
    # Now that rules questions are handled, we can safely match
    # card type mentions as filter requests.
    filter_signals = [
        "show me", "list all", "list the", "which cards", "cards that",
        "cards with", "cards costing", "cards under", "cards over",
        "cards cost", "cards from",
        "cost less", "cost more", "costing", "cheaper than",
        "more expensive", "all cards", "every card",
        "how many cards", "with +", "that give", "that have",
        "what cards",
    ]

    # Card type + list-like intent (but NOT "how do X cards work")
    type_list_signals = [
        "action cards", "treasure cards", "victory cards",
        "attack cards", "duration cards", "reaction cards",
    ]

    # "[Expansion] cards" patterns like "What Seaside cards cost 5?"
    expansion_names = [
        "base", "seaside", "intrigue", "prosperity", "hinterlands",
        "dark ages", "adventures", "empires", "nocturne", "renaissance",
        "menagerie", "alchemy", "cornucopia", "guilds", "allies", "plunder",
    ]
    for exp in expansion_names:
        if f"{exp} cards" in query_lower or f"{exp} card" in query_lower:
            return "filtered_search"

    if any(signal in query_lower for signal in filter_signals):
        return "filtered_search"

    # Type mentions only count as filtered search if combined with
    # list-like words (show, list, which, all, etc.)
    list_intent_words = ["show", "list", "which", "all", "every", "what"]
    if any(t in query_lower for t in type_list_signals):
        if any(w in query_lower for w in list_intent_words):
            return "filtered_search"

    # ── Type 1: Card lookup ──
    # A specific card name + a "tell me about it" signal
    card_lookup_signals = [
        "what does", "what is", "tell me about", "describe",
        "explain", "how does", "what's", "whats",
        "show me the card", "card text",
    ]
    if has_card and any(signal in query_lower for signal in card_lookup_signals):
        return "card_lookup"

    # Also catch bare card name queries like "Chapel" or "Chapel?"
    if has_card and len(query.split()) <= 3:
        return "card_lookup"

    # ── Type 4: Strategy / combo ──
    # This is the catch-all, but also check for explicit strategy signals
    strategy_signals = [
        "combo", "synergy", "synergies", "pairs well",
        "works well", "good with", "best with", "strategy",
        "how to use", "how should i", "when should i",
        "is it worth", "compared to", "vs", "versus",
        "better than", "worse than", "build around",
        "engine", "big money", "rush",
    ]
    if any(signal in query_lower for signal in strategy_signals):
        return "strategy_combo"

    # If a card is mentioned but none of the above matched,
    # it's probably a strategy question about that card
    if has_card:
        return "strategy_combo"

    # Default: if truly ambiguous, try rules first (cheaper than combo)
    return "rules_question"


# ─────────────────────────────────────────────────────────────────
# Filter extraction (for Type 2 queries)
# ─────────────────────────────────────────────────────────────────

def parse_filters(query: str) -> dict:
    """
    Extract structured filters from a natural language query.

    This is a simple keyword-based parser. It handles the most common
    filter patterns. For anything it can't parse, the orchestrator
    can fall back to sending the full query to the LLM.
    """
    import re
    query_lower = query.lower()
    filters = {}

    # Cost filters
    cost_match = re.search(r"cost(?:ing|s)?\s*(?:up to|at most|<=?|under)?\s*(\d+)", query_lower)
    if cost_match:
        filters["max_cost"] = int(cost_match.group(1))

    cost_min = re.search(r"cost(?:ing|s)?\s*(?:at least|>=?|over|more than)\s*(\d+)", query_lower)
    if cost_min:
        filters["min_cost"] = int(cost_min.group(1))

    cost_exact = re.search(r"cost(?:ing|s)?\s*(?:exactly)?\s*(\d+)(?!\s*or)", query_lower)
    if cost_exact and "max_cost" not in filters and "min_cost" not in filters:
        filters["exact_cost"] = int(cost_exact.group(1))

    # Simpler cost patterns: "4 cost", "costing 4 or less"
    simple_cost = re.search(r"(\d+)\s*(?:cost|coins?)", query_lower)
    if simple_cost and not filters:
        filters["max_cost"] = int(simple_cost.group(1))

    or_less = re.search(r"(\d+)\s*or\s*less", query_lower)
    if or_less:
        filters["max_cost"] = int(or_less.group(1))

    or_more = re.search(r"(\d+)\s*or\s*more", query_lower)
    if or_more:
        filters["min_cost"] = int(or_more.group(1))

    # Type filters
    type_map = {
        "action": "Action",
        "treasure": "Treasure",
        "victory": "Victory",
        "attack": "Attack",
        "reaction": "Reaction",
        "duration": "Duration",
        "curse": "Curse",
    }
    for keyword, card_type in type_map.items():
        if keyword in query_lower:
            filters["type"] = card_type
            break

    # Expansion filters
    expansion_keywords = {
        "base": "Base", "seaside": "Seaside", "intrigue": "Intrigue",
        "prosperity": "Prosperity", "hinterlands": "Hinterlands",
        "dark ages": "Dark Ages", "adventures": "Adventures",
        "empires": "Empires", "nocturne": "Nocturne",
        "renaissance": "Renaissance", "menagerie": "Menagerie",
        "alchemy": "Alchemy", "cornucopia": "Cornucopia",
        "guilds": "Guilds", "allies": "Allies", "plunder": "Plunder",
    }
    for keyword, expansion in expansion_keywords.items():
        if keyword in query_lower:
            filters["expansion"] = expansion
            break

    # Attribute filters: "+2 Actions", "give actions", etc.
    actions_match = re.search(r"\+\s*(\d+)\s*actions?", query_lower)
    if actions_match:
        filters["min_plus_actions"] = int(actions_match.group(1))
    elif "give actions" in query_lower or "gives actions" in query_lower:
        filters["min_plus_actions"] = 1

    cards_match = re.search(r"\+\s*(\d+)\s*cards?", query_lower)
    if cards_match:
        filters["min_plus_cards"] = int(cards_match.group(1))
    elif "draw" in query_lower or "draws" in query_lower:
        filters["min_plus_cards"] = 1

    buys_match = re.search(r"\+\s*(\d+)\s*buys?", query_lower)
    if buys_match:
        filters["min_plus_buys"] = int(buys_match.group(1))

    coins_match = re.search(r"\+\s*(\d+)\s*coins?", query_lower)
    if coins_match:
        filters["min_plus_coins"] = int(coins_match.group(1))

    return filters


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        # Type 1: Card lookup
        ("What does Chapel do?", "card_lookup"),
        ("Tell me about Throne Room", "card_lookup"),
        ("Wharf", "card_lookup"),

        # Type 2: Filtered search
        ("Show me all Action cards costing 4 or less", "filtered_search"),
        ("Which cards give +2 Actions?", "filtered_search"),
        ("List all Duration cards", "filtered_search"),
        ("What Seaside cards cost 5?", "filtered_search"),

        # Type 3: Rules question
        ("When can I play Reaction cards?", "rules_question"),
        ("What happens when 3 supply piles are empty?", "rules_question"),
        ("Can I buy multiple cards in one turn?", "rules_question"),

        # Type 4: Strategy / combo
        ("What combos well with Throne Room?", "strategy_combo"),
        ("Is Chapel worth buying early?", "strategy_combo"),
        ("How should I build an engine?", "strategy_combo"),
    ]

    print("── Router Test ────────────────────────────")
    correct = 0
    for query, expected in test_queries:
        result = classify_query(query)
        match = "✅" if result == expected else "❌"
        if result == expected:
            correct += 1
        print(f"  {match} \"{query}\"")
        print(f"       Expected: {expected} | Got: {result}")

    print(f"\n  {correct}/{len(test_queries)} correct")

    # Test filter parsing
    print("\n── Filter Parsing Test ────────────────────")
    filter_tests = [
        "Show me all Action cards costing 4 or less",
        "Which Seaside cards give +2 Actions?",
        "List Duration cards that draw cards",
    ]
    for q in filter_tests:
        filters = parse_filters(q)
        print(f"  \"{q}\"")
        print(f"    → {filters}")
