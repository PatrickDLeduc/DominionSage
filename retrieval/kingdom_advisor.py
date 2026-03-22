"""
kingdom_advisor.py — DominionSage Kingdom Advisor

Takes a set of 10 kingdom cards and generates strategic advice:
  - Opening strategy (first 2 buys)
  - Key card combos to exploit
  - Engine vs Big Money assessment
  - Cards to avoid or deprioritize
  - Attack/defense considerations

Architecture: Reuses the same GPT-4o-mini client and strategy principles
from synthesizer.py. The kingdom card data is formatted as structured
context so the LLM can reason about costs, types, and interactions.
"""

import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: pip install openai")
    sys.exit(1)

from retrieval.synthesizer import STRATEGY_PRINCIPLES


# ─────────────────────────────────────────────────────────────────
# Client (shared singleton)
# ─────────────────────────────────────────────────────────────────

_openai: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai
    if _openai is None:
        if load_dotenv:
            load_dotenv()
        _openai = OpenAI()
    return _openai


# ─────────────────────────────────────────────────────────────────
# Kingdom analysis prompt
# ─────────────────────────────────────────────────────────────────

KINGDOM_SYSTEM = """You are DominionSage's Kingdom Advisor — an expert Dominion strategist.

You are given the 10 kingdom cards for a game. Analyze them and provide
actionable strategy advice. Structure your response with these sections:

## 🎯 Opening Strategy
Recommend the best opening buys (first 2 turns, starting with 3/4 or 5/2 split).
Explain WHY these are the best opening cards for this kingdom.

## 🔗 Key Combos
Identify the 2-3 strongest card synergies in this kingdom. For each combo,
explain the mechanic: what happens when you play them together and why it's
powerful. Be specific (e.g., "Throne Room + Smithy = +6 Cards").

## 📊 Archetype Assessment
Is this kingdom better suited for an Engine, Big Money, Rush, or Slog strategy?
Explain what makes one archetype stronger than the others given these specific cards.

## ⚠️ Cards to Avoid
Are any of the 10 cards traps or low-priority? Explain why they're weak in
THIS specific kingdom (not in general).

## 🛡️ Attack & Defense
If there are Attack cards, how threatening are they? What defenses are available?
If no attacks, note that the game will be more of a "solitaire" race.

RULES:
- Be specific and reference actual card names, costs, and mechanics.
- Keep advice concise — this is a quick pre-game reference, not an essay.
- Use bold for card names when first mentioned.
- Do NOT make up cards or mechanics. Use ONLY the cards provided.
""" + STRATEGY_PRINCIPLES


# ─────────────────────────────────────────────────────────────────
# Core analysis function
# ─────────────────────────────────────────────────────────────────

def format_kingdom_context(cards: list[dict]) -> str:
    """Format the 10 kingdom cards into a structured context block."""
    parts = []
    for i, card in enumerate(cards, 1):
        parts.append(
            f"{i}. {card['name']} — Cost: {card.get('cost', '?')} | "
            f"Type: {card.get('type', '?')} | "
            f"Expansion: {card.get('expansion', '?')}\n"
            f"   Text: {card.get('text', 'N/A')}\n"
            f"   +Actions: {card.get('plus_actions', 0)} | "
            f"+Cards: {card.get('plus_cards', 0)} | "
            f"+Buys: {card.get('plus_buys', 0)} | "
            f"+Coins: {card.get('plus_coins', 0)}"
        )
    return "\n\n".join(parts)


def analyze_kingdom(cards: list[dict]) -> str:
    """
    Analyze a set of kingdom cards and return strategic advice.

    Args:
        cards: List of card dicts (from get_kingdom_cards_by_names).
               Should contain 10 cards for a standard kingdom.

    Returns:
        Markdown-formatted strategy advice string.
    """
    client = _get_client()
    context = format_kingdom_context(cards)

    # Quick kingdom stats for the LLM
    has_village = any(c.get("plus_actions", 0) >= 2 for c in cards)
    has_draw = any(c.get("plus_cards", 0) >= 2 for c in cards)
    has_buy = any(c.get("plus_buys", 0) >= 1 for c in cards)
    has_trash = any(
        "trash" in (c.get("text", "") or "").lower()
        for c in cards
    )
    has_attack = any("Attack" in (c.get("type", "") or "") for c in cards)

    stats = (
        f"Quick stats: "
        f"Village (+2 Actions): {'Yes' if has_village else 'No'} | "
        f"Draw (+2 Cards): {'Yes' if has_draw else 'No'} | "
        f"+Buy: {'Yes' if has_buy else 'No'} | "
        f"Trasher: {'Yes' if has_trash else 'No'} | "
        f"Attack: {'Yes' if has_attack else 'No'}"
    )

    user_message = f"""Here are the 10 kingdom cards for this game:

{context}

{stats}

Analyze this kingdom and provide your strategic advice."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": KINGDOM_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Sorry, I encountered an error analyzing this kingdom: {str(e)}"


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from retrieval.card_lookup import get_kingdom_cards_by_names

    test_kingdom = [
        "Village", "Smithy", "Laboratory", "Festival", "Market",
        "Chapel", "Throne Room", "Witch", "Moat", "Workshop",
    ]

    print("=" * 60)
    print("  Kingdom Advisor — Test Analysis")
    print(f"  Cards: {', '.join(test_kingdom)}")
    print("=" * 60)

    cards = get_kingdom_cards_by_names(test_kingdom)
    print(f"\n  Fetched {len(cards)} cards from DB\n")

    result = analyze_kingdom(cards)
    print(result)
