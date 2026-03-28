"""
card_lookup.py — DominionSage Card Database Queries (Phase 3)

Handles Type 1 (card lookup) and Type 2 (filtered search) queries
by querying the structured `cards` table in Supabase.

This is the "SQL path" of the hybrid retrieval system. It's fast,
precise, and costs $0 per query (no LLM or embedding calls needed).

Analogy: This is like using a WHERE clause in SQL vs. doing a full
text search. When you know the column and value you want, SQL is
always faster and more precise than semantic search.
"""

import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from supabase import create_client, Client
except ImportError:
    print("Missing dependency: pip install supabase")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Client initialization
# ─────────────────────────────────────────────────────────────────

_supabase: Client | None = None


def _get_client() -> Client:
    """Lazy-initialize the Supabase client."""
    global _supabase
    if _supabase is None:
        if load_dotenv:
            load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set.")
        _supabase = create_client(url, key)
    return _supabase


# ─────────────────────────────────────────────────────────────────
# Type 1: Card lookup (by name)
# ─────────────────────────────────────────────────────────────────

def lookup_card(card_name: str) -> list[dict]:
    """
    Look up a specific card by name.

    Uses ILIKE for case-insensitive partial matching, so:
      - "chapel" finds "Chapel"
      - "throne" finds "Throne Room"
      - "village" finds "Village", "Fishing Village", "Native Village"

    For the orchestrator, exact matches are preferred. If multiple
    results come back, the synthesizer will handle disambiguation.
    """
    client = _get_client()

    result = client.table("cards") \
        .select("*") \
        .ilike("name", f"%{card_name}%") \
        .neq("edition", "1e_only") \
        .execute()

    return result.data


def lookup_card_exact(card_name: str) -> dict | None:
    """
    Look up a card by exact name match (case-insensitive).
    Returns a single card dict or None.
    """
    client = _get_client()

    result = client.table("cards") \
        .select("*") \
        .ilike("name", card_name) \
        .neq("edition", "1e_only") \
        .execute()

    return result.data[0] if result.data else None


# ─────────────────────────────────────────────────────────────────
# Type 2: Filtered search
# ─────────────────────────────────────────────────────────────────

def filtered_search(filters: dict) -> list[dict]:
    """
    Search cards with filters extracted by the router's parse_filters().

    Supported filters:
      - max_cost: cards costing at most N
      - min_cost: cards costing at least N
      - exact_cost: cards costing exactly N
      - type: card type contains this string (Action, Treasure, etc.)
      - expansion: exact expansion match
      - min_plus_actions: +Actions >= N
      - min_plus_cards: +Cards >= N
      - min_plus_buys: +Buys >= N
      - min_plus_coins: +Coins >= N

    Automatically excludes:
      - Non-supply cards (Events, Landmarks, Ways, Projects, etc.)
      - Cards with null costs when filtering by cost
    """
    client = _get_client()
    query = client.table("cards").select("*").neq("edition", "1e_only")

    # Exclude non-supply card types from filtered results
    # These are game elements but not cards you buy from supply piles
    non_supply_types = ["Event", "Landmark", "Way", "Project", "Boon",
                        "Hex", "State", "Artifact"]
    for nst in non_supply_types:
        query = query.not_.ilike("type", f"%{nst}%")

    # If filtering by cost, exclude cards with null/missing costs
    has_cost_filter = any(k in filters for k in ["max_cost", "min_cost", "exact_cost"])
    if has_cost_filter:
        query = query.not_.is_("cost", "null")

    if "max_cost" in filters:
        query = query.lte("cost", filters["max_cost"])

    if "min_cost" in filters:
        query = query.gte("cost", filters["min_cost"])

    if "exact_cost" in filters:
        query = query.eq("cost", filters["exact_cost"])

    if "type" in filters:
        query = query.ilike("type", f"%{filters['type']}%")

    if "expansion" in filters:
        query = query.eq("expansion", filters["expansion"])

    if "min_plus_actions" in filters:
        query = query.gte("plus_actions", filters["min_plus_actions"])

    if "min_plus_cards" in filters:
        query = query.gte("plus_cards", filters["min_plus_cards"])

    if "min_plus_buys" in filters:
        query = query.gte("plus_buys", filters["min_plus_buys"])

    if "min_plus_coins" in filters:
        query = query.gte("plus_coins", filters["min_plus_coins"])

    # Order by cost for readability
    query = query.order("cost").order("name")

    result = query.execute()
    return result.data


# ─────────────────────────────────────────────────────────────────
# Kingdom Advisor helpers
# ─────────────────────────────────────────────────────────────────

# Card types that are NOT supply cards (you can't pick these for a kingdom)
_NON_SUPPLY_TYPES = [
    "Event", "Landmark", "Way", "Project", "Boon",
    "Hex", "State", "Artifact",
]

# Basic treasure/victory/curse cards that are always in the game
_ALWAYS_AVAILABLE = {
    "Copper", "Silver", "Gold", "Platinum",
    "Estate", "Duchy", "Province", "Colony",
    "Curse", "Potion",
}


def get_all_kingdom_card_names() -> list[str]:
    """
    Return a sorted list of all supply card names (kingdom-eligible).

    Excludes non-supply types (Events, Landmarks, etc.) and basic
    treasure/victory cards that are always in every game.
    Used to populate the multiselect dropdown in the Kingdom Advisor UI.
    """
    client = _get_client()

    result = client.table("cards").select("name, type").neq("edition", "1e_only").execute()

    names = []
    for card in result.data:
        card_type = card.get("type", "")
        card_name = card.get("name", "")

        # Skip non-supply types
        if any(nst in card_type for nst in _NON_SUPPLY_TYPES):
            continue

        # Skip basic cards
        if card_name in _ALWAYS_AVAILABLE:
            continue

        names.append(card_name)

    return sorted(set(names))


def get_kingdom_cards_by_names(card_names: list[str]) -> list[dict]:
    """
    Batch-fetch full card data for a list of card names.

    Uses exact name matching (case-insensitive). Returns cards in the
    same order they were requested when possible.
    """
    client = _get_client()

    # Supabase .in_() filter for batch lookup
    result = client.table("cards") \
        .select("*") \
        .in_("name", card_names) \
        .neq("edition", "1e_only") \
        .execute()

    # Sort results to match input order
    name_order = {name.lower(): i for i, name in enumerate(card_names)}
    sorted_cards = sorted(
        result.data,
        key=lambda c: name_order.get(c["name"].lower(), 999)
    )

    return sorted_cards


def get_all_expansion_names() -> list[str]:
    """Return a sorted list of distinct expansion names from the cards table."""
    client = _get_client()
    result = client.table("cards").select("expansion").neq("edition", "1e_only").execute()

    expansions = set()
    for card in result.data:
        exp = card.get("expansion")
        if exp:
            expansions.add(exp)

    return sorted(expansions)


def get_random_kingdom(expansions: list[str]) -> list[str]:
    """
    Randomly select 10 kingdom-eligible card names from the given expansions.

    Uses the same filtering logic as get_all_kingdom_card_names() to exclude
    non-supply types and basic cards.
    """
    import random

    client = _get_client()

    # Fetch all cards from the selected expansions
    result = client.table("cards") \
        .select("name, type") \
        .in_("expansion", expansions) \
        .neq("edition", "1e_only") \
        .execute()

    # Apply the same kingdom-eligibility filters
    eligible = []
    for card in result.data:
        card_name = card["name"]
        card_type = card.get("type", "")

        if card_name in _ALWAYS_AVAILABLE:
            continue

        if any(nst in card_type for nst in _NON_SUPPLY_TYPES):
            continue

        eligible.append(card_name)

    # Deduplicate and pick 10
    eligible = list(set(eligible))

    if len(eligible) < 10:
        return sorted(eligible)  # not enough cards

    return sorted(random.sample(eligible, 10))


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Card Lookup Tests ─────────────────────")

    # Type 1: exact lookups
    for name in ["Chapel", "Throne Room", "Wharf"]:
        cards = lookup_card(name)
        if cards:
            c = cards[0]
            print(f"  ✅ {c['name']} | ${c['cost']} | {c['type']} | {c['text'][:60]}...")
        else:
            print(f"  ❌ '{name}' not found")

    print("\n── Filtered Search Tests ──────────────────")

    # Type 2: filtered searches
    tests = [
        ("Action cards costing ≤3", {"type": "Action", "max_cost": 3}),
        ("Cards with +2 Actions", {"min_plus_actions": 2}),
        ("Seaside Duration cards", {"expansion": "Seaside", "type": "Duration"}),
    ]
    for label, filters in tests:
        cards = filtered_search(filters)
        print(f"  {label}: {len(cards)} results")
        for c in cards[:3]:
            print(f"    {c['name']} | ${c['cost']} | {c['type']}")
        if len(cards) > 3:
            print(f"    ... and {len(cards) - 3} more")
