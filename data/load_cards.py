"""
load_cards.py — DominionSage Data Loader (Phase 1)

Reads data/cards.json (output of scrape_cards.py) and upserts all cards
into the Supabase `cards` table.

Prerequisites:
  1. A Supabase project with the `cards` table created (see walkthrough Phase 1.3)
  2. A .env file with SUPABASE_URL and SUPABASE_KEY

Usage:
  python data/load_cards.py               # load all cards
  python data/load_cards.py --dry-run     # preview without writing to DB
  python data/load_cards.py --validate    # load, then run validation queries
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    # python-dotenv is optional — user can export env vars manually
    load_dotenv = None

try:
    from supabase import create_client, Client
except ImportError:
    print("Missing dependency: pip install supabase")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

CARDS_JSON = Path("data/cards.json")

# Fields that go into the DB (must match your table schema)
DB_FIELDS = [
    "name", "cost", "type", "expansion", "text",
    "plus_actions", "plus_cards", "plus_buys", "plus_coins",
]


def get_supabase_client() -> Client:
    """Initialize and return a Supabase client."""
    # Try loading from .env file if python-dotenv is available
    if load_dotenv:
        load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set.")
        print()
        print("Option 1: Create a .env file:")
        print("  SUPABASE_URL=https://your-project.supabase.co")
        print("  SUPABASE_KEY=your-anon-key")
        print()
        print("Option 2: Export environment variables:")
        print("  export SUPABASE_URL=https://your-project.supabase.co")
        print("  export SUPABASE_KEY=your-anon-key")
        sys.exit(1)

    return create_client(url, key)


# ─────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────

def load_cards_from_json() -> list[dict]:
    """Read and validate cards.json."""
    if not CARDS_JSON.exists():
        print(f"Error: {CARDS_JSON} not found.")
        print("Run `python data/scrape_cards.py` first.")
        sys.exit(1)

    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    print(f"Read {len(cards)} cards from {CARDS_JSON}")
    return cards


def prepare_record(card: dict) -> dict:
    """
    Extract only the fields that match the DB schema.
    This protects against extra keys in the JSON breaking the upsert.
    """
    return {field: card.get(field) for field in DB_FIELDS}


def upsert_cards(client: Client, cards: list[dict]) -> dict:
    """
    Upsert all cards into Supabase.

    Uses upsert (INSERT ... ON CONFLICT UPDATE) so this script is
    idempotent — you can run it multiple times safely. If a card with
    the same name already exists, it updates the row instead of failing.

    Returns a summary dict with counts.
    """
    records = [prepare_record(card) for card in cards]
    summary = {"total": len(records), "success": 0, "errors": []}

    # Upsert in batches of 50 (Supabase handles this fine, but batching
    # gives us better error isolation if something goes wrong)
    batch_size = 50
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_label = f"[{i + 1}–{i + len(batch)}]"

        try:
            result = (
                client.table("cards")
                .upsert(batch, on_conflict="name")
                .execute()
            )
            summary["success"] += len(batch)
            print(f"  Upserted batch {batch_label}: {len(batch)} cards")

        except Exception as e:
            summary["errors"].append({"batch": batch_label, "error": str(e)})
            print(f"  Error on batch {batch_label}: {e}")

    return summary


# ─────────────────────────────────────────────────────────────────
# Validation queries (Phase 1.7 from the walkthrough)
# ─────────────────────────────────────────────────────────────────

VALIDATION_QUERIES = [
    {
        "label": "Lookup by name (Chapel)",
        "query": lambda c: c.table("cards").select("name, cost, type, text").eq("name", "Chapel").execute(),
        "check": lambda data: len(data) == 1 and "trash" in data[0]["text"].lower(),
    },
    {
        "label": "Filter by cost (cost <= 3)",
        "query": lambda c: c.table("cards").select("name, cost").lte("cost", 3).execute(),
        "check": lambda data: len(data) >= 10 and all(d["cost"] <= 3 for d in data),
    },
    {
        "label": "Filter by expansion (Seaside)",
        "query": lambda c: c.table("cards").select("name, expansion").eq("expansion", "Seaside").execute(),
        "check": lambda data: len(data) >= 20 and all(d["expansion"] == "Seaside" for d in data),
    },
    {
        "label": "Filter by type (Action)",
        "query": lambda c: c.table("cards").select("name, type").ilike("type", "%Action%").execute(),
        "check": lambda data: len(data) >= 30,
    },
    {
        "label": "Filter by attribute (+Actions >= 2)",
        "query": lambda c: c.table("cards").select("name, plus_actions").gte("plus_actions", 2).execute(),
        "check": lambda data: len(data) >= 3 and all(d["plus_actions"] >= 2 for d in data),
    },
]


def run_validation(client: Client) -> None:
    """Run the 5 validation queries from Phase 1.7 of the walkthrough."""
    print("\n── Validation Queries ─────────────────────")
    passed = 0
    failed = 0

    for vq in VALIDATION_QUERIES:
        try:
            result = vq["query"](client)
            data = result.data

            if vq["check"](data):
                print(f"  ✅ {vq['label']} — {len(data)} rows")
                passed += 1
            else:
                print(f"  ❌ {vq['label']} — {len(data)} rows (check failed)")
                failed += 1

        except Exception as e:
            print(f"  ❌ {vq['label']} — error: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("  🎉 All validation queries passed! Phase 1 checkpoint complete.")
    else:
        print("  ⚠️  Some queries failed. Check your table schema and data.")


# ─────────────────────────────────────────────────────────────────
# Dry run
# ─────────────────────────────────────────────────────────────────

def dry_run(cards: list[dict]) -> None:
    """Preview what would be loaded without touching the database."""
    print("\n── Dry Run (no DB writes) ─────────────────")

    # Group by expansion
    by_expansion = {}
    for card in cards:
        exp = card.get("expansion", "Unknown")
        by_expansion.setdefault(exp, []).append(card)

    for exp, exp_cards in sorted(by_expansion.items()):
        print(f"\n  {exp} ({len(exp_cards)} cards):")
        for card in exp_cards:
            attrs = []
            if card.get("plus_actions"): attrs.append(f"+{card['plus_actions']}A")
            if card.get("plus_cards"):   attrs.append(f"+{card['plus_cards']}C")
            if card.get("plus_buys"):    attrs.append(f"+{card['plus_buys']}B")
            if card.get("plus_coins"):   attrs.append(f"+{card['plus_coins']}$")
            attr_str = " ".join(attrs) if attrs else "—"
            print(f"    {card['name']:20} | ${card['cost']} | {card['type']:25} | {attr_str}")

    print(f"\n  Total: {len(cards)} cards would be upserted.")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load Dominion card data into Supabase."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview cards without writing to the database.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation queries after loading.",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("DominionSage — Card Loader")
    print("=" * 50)

    # Load cards from JSON
    cards = load_cards_from_json()

    if args.dry_run:
        dry_run(cards)
        return

    # Connect to Supabase
    print("\nConnecting to Supabase...")
    client = get_supabase_client()
    print("  Connected.")

    # Upsert cards
    print(f"\nUpserting {len(cards)} cards...")
    summary = upsert_cards(client, cards)

    # Report
    print(f"\n── Load Summary ──────────────────────────")
    print(f"  Total:    {summary['total']}")
    print(f"  Success:  {summary['success']}")
    print(f"  Errors:   {len(summary['errors'])}")

    if summary["errors"]:
        print("\n  Error details:")
        for err in summary["errors"]:
            print(f"    {err['batch']}: {err['error']}")

    # Validate if requested
    if args.validate:
        run_validation(client)
    else:
        print("\n  Tip: run with --validate to verify the data in Supabase.")

    print()


if __name__ == "__main__":
    main()
