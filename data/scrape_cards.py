"""
scrape_cards.py — DominionSage Data Ingestion (Phase 1)

Fetches Dominion card data for the Base (2nd Ed) and Seaside (2nd Ed) expansions.

Strategy:
  1. Try the Dominion Strategy Wiki Cargo API (structured data export)
  2. If that fails (bot protection, downtime, etc.), fall back to the curated
     local dataset below — hand-verified against the official card lists.

Output:
  data/cards.json — ready for load_cards.py to push into Supabase.

Usage:
  python data/scrape_cards.py
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Wiki Cargo API scraper
# ─────────────────────────────────────────────────────────────────

CARGO_API = "https://wiki.dominionstrategy.com/api.php"

def try_wiki_cargo(expansion: str) -> list[dict] | None:
    """
    Attempt to pull card data from the Dominion Strategy Wiki's Cargo tables.
    Returns a list of card dicts, or None if the request fails.
    """
    params = {
        "action": "cargoquery",
        "tables": "CardData",
        "fields": "Name,Set,Types,Cost,Text,Actions,Cards,Buys,Coins",
        "where": f'Set="{expansion}"',
        "format": "json",
        "limit": "100",
    }
    try:
        print(f"  Trying wiki Cargo API for '{expansion}'...")
        resp = requests.get(CARGO_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("cargoquery", [])
        if not results:
            print(f"  No results from Cargo API for '{expansion}'.")
            return None

        cards = []
        for item in results:
            row = item.get("title", {})
            cards.append(parse_cargo_row(row, expansion))

        print(f"  Got {len(cards)} cards from wiki for '{expansion}'.")
        return cards

    except Exception as e:
        print(f"  Wiki API failed: {e}")
        return None


def parse_cargo_row(row: dict, expansion: str) -> dict:
    """Convert a Cargo API row into our standard card dict."""
    return {
        "name": row.get("Name", "").strip(),
        "cost": safe_int(row.get("Cost", "")),
        "type": row.get("Types", "").strip(),
        "expansion": expansion,
        "text": row.get("Text", "").strip(),
        "plus_actions": safe_int(row.get("Actions", "0")),
        "plus_cards": safe_int(row.get("Cards", "0")),
        "plus_buys": safe_int(row.get("Buys", "0")),
        "plus_coins": safe_int(row.get("Coins", "0")),
    }


def safe_int(value) -> int:
    """Parse an integer from a string, returning 0 for anything unparseable."""
    if value is None:
        return 0
    try:
        # Handle strings like "2" or "2P" (potion costs)
        return int(re.sub(r"[^\d]", "", str(value)) or "0")
    except (ValueError, TypeError):
        return 0


# ─────────────────────────────────────────────────────────────────
# Curated fallback dataset
# Hand-verified against official Dominion 2nd Edition card lists
# ─────────────────────────────────────────────────────────────────

def parse_attributes(text: str) -> dict:
    """
    Extract +Actions, +Cards, +Buys, +Coins from card text.
    This is the 'compute at ingest time' pattern from the design doc.
    """
    attrs = {
        "plus_actions": 0,
        "plus_cards": 0,
        "plus_buys": 0,
        "plus_coins": 0,
    }
    if not text:
        return attrs

    # Match patterns like "+2 Actions", "+1 Card", "+1 Buy", "+3 Coins"
    patterns = {
        "plus_actions": r"\+(\d+)\s+Actions?",
        "plus_cards": r"\+(\d+)\s+Cards?",
        "plus_buys": r"\+(\d+)\s+Buys?",
        "plus_coins": r"\+(\d+)\s+Coins?",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            attrs[key] = int(match.group(1))

    return attrs


# ── Base Set (2nd Edition) ──────────────────────────────────────

BASE_CARDS = [
    # Kingdom cards
    {
        "name": "Cellar",
        "cost": 2,
        "type": "Action",
        "text": "+1 Action. Discard any number of cards, then draw that many.",
    },
    {
        "name": "Chapel",
        "cost": 2,
        "type": "Action",
        "text": "Trash up to 4 cards from your hand.",
    },
    {
        "name": "Moat",
        "cost": 2,
        "type": "Action - Reaction",
        "text": "+2 Cards. When another player plays an Attack card, you may first reveal this from your hand, to be unaffected by it.",
    },
    {
        "name": "Harbinger",
        "cost": 3,
        "type": "Action",
        "text": "+1 Card. +1 Action. Look through your discard pile. You may put a card from it onto your deck.",
    },
    {
        "name": "Merchant",
        "cost": 3,
        "type": "Action",
        "text": "+1 Card. +1 Action. The first time you play a Silver this turn, +1 Coin.",
    },
    {
        "name": "Vassal",
        "cost": 3,
        "type": "Action",
        "text": "+2 Coins. Discard the top card of your deck. If it's an Action card, you may play it.",
    },
    {
        "name": "Village",
        "cost": 3,
        "type": "Action",
        "text": "+1 Card. +2 Actions.",
    },
    {
        "name": "Workshop",
        "cost": 3,
        "type": "Action",
        "text": "Gain a card costing up to 4 Coins.",
    },
    {
        "name": "Bureaucrat",
        "cost": 4,
        "type": "Action - Attack",
        "text": "Gain a Silver onto your deck. Each other player reveals a Victory card from their hand and puts it onto their deck (or reveals a hand with no Victory cards).",
    },
    {
        "name": "Gardens",
        "cost": 4,
        "type": "Victory",
        "text": "Worth 1 Victory Point per 10 cards you have (round down).",
    },
    {
        "name": "Militia",
        "cost": 4,
        "type": "Action - Attack",
        "text": "+2 Coins. Each other player discards down to 3 cards in hand.",
    },
    {
        "name": "Moneylender",
        "cost": 4,
        "type": "Action",
        "text": "You may trash a Copper from your hand for +3 Coins.",
    },
    {
        "name": "Poacher",
        "cost": 4,
        "type": "Action",
        "text": "+1 Card. +1 Action. +1 Coin. Discard a card per empty Supply pile.",
    },
    {
        "name": "Remodel",
        "cost": 4,
        "type": "Action",
        "text": "Trash a card from your hand. Gain a card costing up to 2 Coins more than it.",
    },
    {
        "name": "Smithy",
        "cost": 4,
        "type": "Action",
        "text": "+3 Cards.",
    },
    {
        "name": "Throne Room",
        "cost": 4,
        "type": "Action",
        "text": "You may play an Action card from your hand twice.",
    },
    {
        "name": "Bandit",
        "cost": 5,
        "type": "Action - Attack",
        "text": "Gain a Gold. Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper, and discards the rest.",
    },
    {
        "name": "Council Room",
        "cost": 5,
        "type": "Action",
        "text": "+4 Cards. +1 Buy. Each other player draws a card.",
    },
    {
        "name": "Festival",
        "cost": 5,
        "type": "Action",
        "text": "+2 Actions. +1 Buy. +2 Coins.",
    },
    {
        "name": "Laboratory",
        "cost": 5,
        "type": "Action",
        "text": "+2 Cards. +1 Action.",
    },
    {
        "name": "Library",
        "cost": 5,
        "type": "Action",
        "text": "Draw until you have 7 cards in hand, skipping any Action cards you choose to; set those aside, discarding them afterwards.",
    },
    {
        "name": "Market",
        "cost": 5,
        "type": "Action",
        "text": "+1 Card. +1 Action. +1 Buy. +1 Coin.",
    },
    {
        "name": "Mine",
        "cost": 5,
        "type": "Action",
        "text": "You may trash a Treasure from your hand. Gain a Treasure to your hand costing up to 3 Coins more than it.",
    },
    {
        "name": "Sentry",
        "cost": 5,
        "type": "Action",
        "text": "+1 Card. +1 Action. Look at the top 2 cards of your deck. Trash and/or discard any number of them. Put the rest back on top in any order.",
    },
    {
        "name": "Witch",
        "cost": 5,
        "type": "Action - Attack",
        "text": "+2 Cards. Each other player gains a Curse.",
    },
    {
        "name": "Artisan",
        "cost": 6,
        "type": "Action",
        "text": "Gain a card to your hand costing up to 5 Coins. Put a card from your hand onto your deck.",
    },
    # Basic Supply cards (included for completeness — useful for rules questions)
    {
        "name": "Copper",
        "cost": 0,
        "type": "Treasure",
        "text": "1 Coin.",
    },
    {
        "name": "Silver",
        "cost": 3,
        "type": "Treasure",
        "text": "2 Coins.",
    },
    {
        "name": "Gold",
        "cost": 6,
        "type": "Treasure",
        "text": "3 Coins.",
    },
    {
        "name": "Estate",
        "cost": 2,
        "type": "Victory",
        "text": "1 Victory Point.",
    },
    {
        "name": "Duchy",
        "cost": 5,
        "type": "Victory",
        "text": "3 Victory Points.",
    },
    {
        "name": "Province",
        "cost": 8,
        "type": "Victory",
        "text": "6 Victory Points.",
    },
    {
        "name": "Curse",
        "cost": 0,
        "type": "Curse",
        "text": "-1 Victory Point.",
    },
]


# ── Seaside (2nd Edition) ──────────────────────────────────────

SEASIDE_CARDS = [
    {
        "name": "Haven",
        "cost": 2,
        "type": "Action - Duration",
        "text": "+1 Card. +1 Action. Set aside a card from your hand face down (under this). At the start of your next turn, put it into your hand.",
    },
    {
        "name": "Lighthouse",
        "cost": 2,
        "type": "Action - Duration",
        "text": "+1 Action. Now and at the start of your next turn: +1 Coin. Between now and then, when another player plays an Attack card, it doesn't affect you.",
    },
    {
        "name": "Native Village",
        "cost": 2,
        "type": "Action",
        "text": "+2 Actions. Choose one: Set aside the top card of your deck face down on your Native Village mat; or put all the cards from your mat into your hand.",
    },
    {
        "name": "Astrolabe",
        "cost": 3,
        "type": "Treasure - Duration",
        "text": "Now and at the start of your next turn: +1 Coin and +1 Buy.",
    },
    {
        "name": "Fishing Village",
        "cost": 3,
        "type": "Action - Duration",
        "text": "+2 Actions. +1 Coin. At the start of your next turn: +1 Action and +1 Coin.",
    },
    {
        "name": "Lookout",
        "cost": 3,
        "type": "Action",
        "text": "+1 Action. Look at the top 3 cards of your deck. Trash one of them. Discard one of them. Put the other one on top of your deck.",
    },
    {
        "name": "Monkey",
        "cost": 3,
        "type": "Action - Duration",
        "text": "Until your next turn, when the player to your right gains a card, +1 Card. At the start of your next turn, +1 Card.",
    },
    {
        "name": "Sea Chart",
        "cost": 3,
        "type": "Action",
        "text": "+1 Card. +1 Action. Reveal the top card of your deck. If you have a copy of it in play, put it into your hand.",
    },
    {
        "name": "Smugglers",
        "cost": 3,
        "type": "Action",
        "text": "Gain a copy of a card the player to your right gained on their last turn, costing up to 6 Coins.",
    },
    {
        "name": "Warehouse",
        "cost": 3,
        "type": "Action",
        "text": "+3 Cards. +1 Action. Discard 3 cards.",
    },
    {
        "name": "Blockade",
        "cost": 4,
        "type": "Action - Duration - Attack",
        "text": "Gain a card costing up to 4 Coins, setting it aside. At the start of your next turn, put it into your hand. While it's set aside, when another player gains a copy of it on their turn, they also gain a Curse.",
    },
    {
        "name": "Caravan",
        "cost": 4,
        "type": "Action - Duration",
        "text": "+1 Card. +1 Action. At the start of your next turn, +1 Card.",
    },
    {
        "name": "Cutpurse",
        "cost": 4,
        "type": "Action - Attack",
        "text": "+2 Coins. Each other player discards a Copper card (or reveals a hand with no Copper).",
    },
    {
        "name": "Island",
        "cost": 4,
        "type": "Action - Victory",
        "text": "Put this and a card from your hand onto your Island mat. 2 Victory Points.",
    },
    {
        "name": "Sailor",
        "cost": 4,
        "type": "Action - Duration",
        "text": "+1 Action. Once this turn, when you gain a Duration card, you may play it. At the start of your next turn, +2 Coins, and you may trash a card from your hand.",
    },
    {
        "name": "Salvager",
        "cost": 4,
        "type": "Action",
        "text": "+1 Buy. Trash a card from your hand. +Coins equal to its cost.",
    },
    {
        "name": "Tide Pools",
        "cost": 4,
        "type": "Action - Duration",
        "text": "+3 Cards. +1 Action. At the start of your next turn, discard 2 cards.",
    },
    {
        "name": "Treasure Map",
        "cost": 4,
        "type": "Action",
        "text": "Trash this and a Treasure Map from your hand. If you trashed two Treasure Maps, gain 4 Golds onto your deck.",
    },
    {
        "name": "Bazaar",
        "cost": 5,
        "type": "Action",
        "text": "+1 Card. +2 Actions. +1 Coin.",
    },
    {
        "name": "Corsair",
        "cost": 5,
        "type": "Action - Duration - Attack",
        "text": "+2 Coins. At the start of your next turn, +1 Card. Until then, each other player trashes the first Silver or Gold they play each turn.",
    },
    {
        "name": "Merchant Ship",
        "cost": 5,
        "type": "Action - Duration",
        "text": "Now and at the start of your next turn: +2 Coins.",
    },
    {
        "name": "Pirate",
        "cost": 5,
        "type": "Action - Duration - Reaction",
        "text": "At the start of your next turn, gain a Treasure costing up to 6 Coins to your hand. When any player gains a Treasure, you may play this from your hand.",
    },
    {
        "name": "Sea Witch",
        "cost": 5,
        "type": "Action - Duration - Attack",
        "text": "+2 Cards. Each other player gains a Curse. At the start of your next turn, +2 Cards, then discard 2 cards.",
    },
    {
        "name": "Tactician",
        "cost": 5,
        "type": "Action - Duration",
        "text": "If you have at least one card in hand: Discard your hand, and at the start of your next turn, +5 Cards, +1 Action, and +1 Buy.",
    },
    {
        "name": "Treasury",
        "cost": 5,
        "type": "Action",
        "text": "+1 Card. +1 Action. +1 Coin. When you discard this from play, if you didn't gain a Victory card this turn, you may put this onto your deck.",
    },
    {
        "name": "Wharf",
        "cost": 5,
        "type": "Action - Duration",
        "text": "Now and at the start of your next turn: +2 Cards and +1 Buy.",
    },
]


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def build_card_list() -> list[dict]:
    """
    Build the full card list. Try the wiki API first for each expansion;
    fall back to the curated local data if that fails.
    """
    all_cards = []

    for expansion, fallback in [("Dominion", BASE_CARDS), ("Seaside", SEASIDE_CARDS)]:
        wiki_cards = try_wiki_cargo(expansion)

        if wiki_cards and len(wiki_cards) >= 5:
            # Wiki data looks good — use it
            all_cards.extend(wiki_cards)
        else:
            # Fall back to curated local data
            print(f"  Using curated local data for '{expansion}' ({len(fallback)} cards).")
            for card in fallback:
                attrs = parse_attributes(card["text"])
                all_cards.append({
                    "name": card["name"],
                    "cost": card["cost"],
                    "type": card["type"],
                    "expansion": expansion if expansion != "Dominion" else "Base",
                    "text": card["text"],
                    **attrs,
                })

    return all_cards


def validate_cards(cards: list[dict]) -> None:
    """Run basic validation checks on the card list."""
    names = set()
    issues = []

    for i, card in enumerate(cards):
        # Required fields
        for field in ["name", "cost", "type", "expansion", "text"]:
            if not card.get(field) and card.get(field) != 0:
                issues.append(f"  Card #{i}: missing '{field}' — {card.get('name', '???')}")

        # Duplicate names
        if card["name"] in names:
            issues.append(f"  Duplicate card name: {card['name']}")
        names.add(card["name"])

        # Sanity check on numeric fields
        for field in ["plus_actions", "plus_cards", "plus_buys", "plus_coins"]:
            val = card.get(field, 0)
            if not isinstance(val, int) or val < 0 or val > 10:
                issues.append(f"  Card '{card['name']}': suspicious {field}={val}")

    if issues:
        print(f"\n⚠️  Validation found {len(issues)} issue(s):")
        for issue in issues:
            print(issue)
    else:
        print(f"\n✅ Validation passed: {len(cards)} cards, 0 issues.")


def print_summary(cards: list[dict]) -> None:
    """Print a quick summary table."""
    expansions = {}
    for card in cards:
        exp = card["expansion"]
        expansions[exp] = expansions.get(exp, 0) + 1

    print("\n── Card Summary ──────────────────────────")
    for exp, count in sorted(expansions.items()):
        print(f"  {exp}: {count} cards")
    print(f"  Total: {len(cards)} cards")

    # Show a sample card for sanity checking
    sample = next((c for c in cards if c["name"] == "Village"), cards[0])
    print(f"\n── Sample Card ───────────────────────────")
    for k, v in sample.items():
        print(f"  {k}: {v}")


def main():
    print("=" * 50)
    print("DominionSage — Card Data Scraper")
    print("=" * 50)

    # Ensure output directory exists
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cards.json"

    # Build card list
    cards = build_card_list()

    # Validate
    validate_cards(cards)

    # Summary
    print_summary(cards)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved {len(cards)} cards to {output_path}")
    print("   Next step: python data/load_cards.py")


if __name__ == "__main__":
    main()
