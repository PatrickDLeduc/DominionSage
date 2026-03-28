"""
tools.py — OpenAI function-calling tool definitions for the DominionSage agent.

Each tool wraps an existing retrieval function and returns two things:
  1. A text summary for the LLM conversation (what the agent sees)
  2. A list of source dicts for the UI's citation panel (accumulated silently)

The TOOL_SCHEMAS list is passed directly to the OpenAI chat completions API.
"""

import json
from retrieval.card_lookup import lookup_card, filtered_search
from retrieval.hybrid_search import hybrid_search


# ─────────────────────────────────────────────────────────────────
# OpenAI tool schemas
# ─────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_card",
            "description": (
                "Look up a specific Dominion card by name. Returns card data "
                "including cost, type, text, expansion, and stats (+Actions, "
                "+Cards, +Buys, +Coins). Use for questions about what a specific "
                "card does or its properties."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "card_name": {
                        "type": "string",
                        "description": "The card name to look up (case-insensitive, partial match supported)",
                    },
                },
                "required": ["card_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cards",
            "description": (
                "Search the card database with structured filters. Use for "
                "questions like 'show me all Duration cards costing 4 or less' "
                "or 'which cards give +2 Actions'. Returns matching cards sorted "
                "by cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_cost": {
                        "type": "integer",
                        "description": "Maximum card cost (inclusive)",
                    },
                    "min_cost": {
                        "type": "integer",
                        "description": "Minimum card cost (inclusive)",
                    },
                    "exact_cost": {
                        "type": "integer",
                        "description": "Exact card cost",
                    },
                    "type": {
                        "type": "string",
                        "description": "Card type to filter by (e.g. Action, Treasure, Victory, Attack, Reaction, Duration)",
                    },
                    "expansion": {
                        "type": "string",
                        "description": "Expansion name to filter by (e.g. Base, Seaside, Prosperity)",
                    },
                    "min_plus_actions": {
                        "type": "integer",
                        "description": "Minimum +Actions value",
                    },
                    "min_plus_cards": {
                        "type": "integer",
                        "description": "Minimum +Cards value",
                    },
                    "min_plus_buys": {
                        "type": "integer",
                        "description": "Minimum +Buys value",
                    },
                    "min_plus_coins": {
                        "type": "integer",
                        "description": "Minimum +Coins value",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": (
                "Search Dominion rulebooks and strategy guides using semantic "
                "and keyword search. Use for rules questions, timing questions, "
                "or when you need to look up specific game mechanics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what rules or mechanics to find",
                    },
                    "expansion": {
                        "type": "string",
                        "description": "Optional expansion to limit the search to (e.g. Base, Seaside)",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_strategy",
            "description": (
                "Search for strategy advice, card combos, and synergies. "
                "Optionally provide a specific card name to also retrieve that "
                "card's full data alongside strategy content. Use for combo "
                "questions, archetype advice, or 'what works well with X'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The strategy question or topic to search for",
                    },
                    "card_name": {
                        "type": "string",
                        "description": "Optional specific card name to also look up alongside strategy results",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────
# Tool executors
# ─────────────────────────────────────────────────────────────────

def _format_card(card: dict) -> str:
    """Format a single card dict into a readable text block."""
    return (
        f"  {card['name']} | Cost: {card.get('cost', '?')} | Type: {card.get('type', '?')} "
        f"| Expansion: {card.get('expansion', '?')}\n"
        f"  Text: {card.get('text', 'N/A')}\n"
        f"  +Actions: {card.get('plus_actions', 0)} | +Cards: {card.get('plus_cards', 0)} "
        f"| +Buys: {card.get('plus_buys', 0)} | +Coins: {card.get('plus_coins', 0)}"
    )


def _format_chunk(chunk: dict) -> str:
    """Format a single rulebook/strategy chunk into a readable text block."""
    expansion = chunk.get("expansion", "?")
    page = chunk.get("source_page", "?")
    chunk_type = chunk.get("chunk_type", "")
    card_name = chunk.get("card_name", "")
    header = f"{chunk_type}: {card_name}" if chunk_type and card_name else f"{expansion} p.{page}"
    return f"  [{header}]\n  {chunk.get('chunk_text', 'N/A')}"


def execute_tool(name: str, arguments: dict) -> tuple[str, list[dict]]:
    """
    Execute a tool by name with the given arguments.

    Returns:
        (text_for_llm, sources) where text_for_llm is what the agent sees
        and sources is a list of source dicts for the UI citation panel.
    """
    if name == "lookup_card":
        cards = lookup_card(arguments["card_name"])
        if not cards:
            return f"No cards found matching '{arguments['card_name']}'.", []
        text = f"Found {len(cards)} card(s):\n\n" + "\n\n".join(_format_card(c) for c in cards)
        sources = [{"type": "card_db", "data": c} for c in cards]
        return text, sources

    elif name == "search_cards":
        filters = {k: v for k, v in arguments.items() if v is not None}
        cards = filtered_search(filters)
        if not cards:
            return "No cards matched the given filters.", []
        total = len(cards)
        display_cards = cards[:20]
        text = f"Found {total} matching card(s):\n\n" + "\n\n".join(_format_card(c) for c in display_cards)
        if total > 20:
            text += f"\n\n(Showing 20 of {total}. Narrow your filters for more specific results.)"
        sources = [{"type": "card_db", "data": c} for c in display_cards]
        if total > 20:
            sources.append({
                "type": "meta",
                "data": {
                    "note": f"Showing 20 of {total} matching cards. "
                            "Try narrowing your search with a cost range, "
                            "expansion, or card type to see more specific results."
                },
            })
        return text, sources

    elif name == "search_rules":
        query = arguments["query"]
        expansion = arguments.get("expansion")
        chunks = hybrid_search(query, top_k=5, expansion=expansion)
        if not chunks:
            return "No relevant rules or mechanics found.", []
        text = f"Found {len(chunks)} relevant rulebook section(s):\n\n" + "\n\n".join(_format_chunk(c) for c in chunks)
        sources = [{"type": "rulebook", "data": c} for c in chunks]
        return text, sources

    elif name == "search_strategy":
        query = arguments["query"]
        card_name = arguments.get("card_name")
        sources = []
        text_parts = []

        if card_name:
            cards = lookup_card(card_name)
            if cards:
                text_parts.append(f"Card data for '{card_name}':\n\n" + "\n\n".join(_format_card(c) for c in cards))
                sources.extend([{"type": "card_db", "data": c} for c in cards])

        chunks = hybrid_search(query, top_k=3 if card_name else 5)
        if chunks:
            text_parts.append(f"Strategy content:\n\n" + "\n\n".join(_format_chunk(c) for c in chunks))
            sources.extend([{"type": "rulebook", "data": c} for c in chunks])

        if not text_parts:
            return "No strategy content found.", []
        return "\n\n---\n\n".join(text_parts), sources

    else:
        return f"Unknown tool: {name}", []
