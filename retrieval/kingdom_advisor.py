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
    from openai import OpenAI, LengthFinishReasonError
except ImportError:
    print("Missing dependency: pip install openai")
    sys.exit(1)

from retrieval.synthesizer import STRATEGY_PRINCIPLES
from retrieval.models import KingdomAdvisorResponse


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
actionable strategy advice.

RULES:
- Be specific and reference actual card names, costs, and mechanics.
- Keep advice concise — this is a quick pre-game reference, not an essay.
- Do NOT make up cards or mechanics. Use ONLY the cards provided.
- For opening strategy, explain WHY these are the best buys for this kingdom.
- For combos, explain the mechanic (e.g., "Throne Room + Smithy = +6 Cards").
- For cards to avoid, explain why they're weak in THIS specific kingdom.
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


def analyze_kingdom(cards: list[dict]) -> KingdomAdvisorResponse:
    """
    Analyze a set of kingdom cards and return strategic advice.

    Args:
        cards: List of card dicts (from get_kingdom_cards_by_names).
               Should contain 10 cards for a standard kingdom.

    Returns:
        KingdomAdvisorResponse with structured sections for opening
        strategy, combos, archetype assessment, cards to avoid, and
        attack/defense analysis.
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
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": KINGDOM_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=1500,
            response_format=KingdomAdvisorResponse,
        )

        if response.choices[0].message.refusal:
            return KingdomAdvisorResponse(
                opening_strategy={"three_four_split": "Unable to analyze.", "five_two_split": "Unable to analyze."},
                key_combos=[],
                archetype_assessment="The model declined to analyze this kingdom.",
                cards_to_avoid=[],
                attack_and_defense="",
            )

        return response.choices[0].message.parsed

    except LengthFinishReasonError:
        return KingdomAdvisorResponse(
            opening_strategy={"three_four_split": "Analysis too long.", "five_two_split": "Analysis too long."},
            key_combos=[],
            archetype_assessment="Response was too long to complete. Try again.",
            cards_to_avoid=[],
            attack_and_defense="",
        )
    except Exception as e:
        return KingdomAdvisorResponse(
            opening_strategy={"three_four_split": "Error occurred.", "five_two_split": "Error occurred."},
            key_combos=[],
            archetype_assessment=f"Sorry, I encountered an error analyzing this kingdom: {str(e)}",
            cards_to_avoid=[],
            attack_and_defense="",
        )


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
