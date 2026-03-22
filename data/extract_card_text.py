"""
extract_card_text.py — Extract card text from rulebook chunks

For cards that are missing their card text (the 537 from the xlsx import),
this script searches the existing rulebook_chunks in Supabase for chunks
that mention each card by name, then uses GPT-4o-mini to extract the
actual card text from those chunks.

This is a neat trick: you already did the hard work of downloading,
chunking, and embedding 17 rulebooks. Now we're mining that data
for structured card descriptions — turning unstructured text back
into structured data.

Usage:
  python data/extract_card_text.py                # extract all missing text
  python data/extract_card_text.py --dry-run      # preview which cards need text
  python data/extract_card_text.py --limit 10     # only process 10 cards (for testing)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: pip install openai")
    sys.exit(1)

try:
    from supabase import create_client, Client
except ImportError:
    print("Missing dependency: pip install supabase")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

CARDS_JSON = Path("data/cards.json")
EMBEDDING_MODEL = "text-embedding-3-small"

EXTRACTION_PROMPT = """You are extracting Dominion card game text from rulebook excerpts.

Given the card name and rulebook excerpt(s) below, extract ONLY the card's
official text — the text that would be printed on the physical card.

Rules:
- Return ONLY the card text, nothing else
- Include +Actions, +Cards, +Buys, +Coins if mentioned
- Do NOT include strategy tips, clarifications, or FAQ text
- Do NOT include the card name, cost, or type — just the card text
- If the card text is not found in the excerpts, respond with exactly: NOT_FOUND
- Keep it concise — card text is usually 1-3 sentences

Example output for Village: "+1 Card. +2 Actions."
Example output for Chapel: "Trash up to 4 cards from your hand."
"""


# ─────────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────────

def init_clients() -> tuple[OpenAI, Client]:
    if load_dotenv:
        load_dotenv()

    openai_client = OpenAI()
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY"),
    )
    return openai_client, supabase


# ─────────────────────────────────────────────────────────────────
# Search for card mentions in rulebook chunks
# ─────────────────────────────────────────────────────────────────

def find_chunks_mentioning_card(
    supabase: Client,
    openai_client: OpenAI,
    card_name: str,
    expansion: str,
) -> list[dict]:
    """
    Find rulebook chunks that mention a specific card.

    Two strategies:
    1. First try a text search (faster, no API cost)
    2. Fall back to semantic search if text search finds nothing
    """
    # Strategy 1: Text search — look for the card name in chunk_text
    # This is the fast, free path
    try:
        result = supabase.table("rulebook_chunks") \
            .select("chunk_text, source_page, expansion") \
            .ilike("chunk_text", f"%{card_name}%") \
            .eq("expansion", expansion) \
            .limit(5) \
            .execute()

        if result.data:
            return result.data
    except Exception:
        pass

    # Also try without expansion filter (card might be in base rulebook)
    try:
        result = supabase.table("rulebook_chunks") \
            .select("chunk_text, source_page, expansion") \
            .ilike("chunk_text", f"%{card_name}%") \
            .limit(5) \
            .execute()

        if result.data:
            return result.data
    except Exception:
        pass

    # Strategy 2: Semantic search — embed the card name and find similar chunks
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=f"{card_name} card text description",
        )
        query_embedding = response.data[0].embedding

        result = supabase.rpc("match_chunks_by_expansion", {
            "query_embedding": query_embedding,
            "match_count": 3,
            "filter_expansion": expansion,
        }).execute()

        if result.data:
            return result.data
    except Exception:
        pass

    return []


# ─────────────────────────────────────────────────────────────────
# Extract card text using LLM
# ─────────────────────────────────────────────────────────────────

def extract_card_text(
    openai_client: OpenAI,
    card_name: str,
    chunks: list[dict],
) -> str | None:
    """
    Use GPT-4o-mini to extract the card text from rulebook chunks.
    Returns the card text string, or None if not found.
    """
    # Combine chunk texts
    chunk_texts = []
    for chunk in chunks[:5]:  # Max 5 chunks to keep context manageable
        text = chunk.get("chunk_text", "")
        page = chunk.get("source_page", "?")
        exp = chunk.get("expansion", "?")
        chunk_texts.append(f"[{exp} p.{page}]: {text}")

    context = "\n\n---\n\n".join(chunk_texts)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Card name: {card_name}\n\nRulebook excerpts:\n{context}"},
            ],
            temperature=0.1,  # Very low — we want factual extraction
            max_tokens=200,
        )
        result = response.choices[0].message.content.strip()

        if result == "NOT_FOUND" or not result:
            return None

        return result

    except Exception as e:
        print(f"      LLM error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Attribute extraction
# ─────────────────────────────────────────────────────────────────

def parse_attributes(text: str) -> dict:
    """Extract +Actions, +Cards, +Buys, +Coins from card text."""
    import re
    attrs = {
        "plus_actions": 0,
        "plus_cards": 0,
        "plus_buys": 0,
        "plus_coins": 0,
    }
    if not text:
        return attrs

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


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract card text from rulebook chunks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show which cards need text without extracting.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many cards (for testing).")
    args = parser.parse_args()

    print("=" * 50)
    print("DominionSage — Card Text Extractor")
    print("=" * 50)

    # Load cards
    if not CARDS_JSON.exists():
        print(f"Error: {CARDS_JSON} not found. Run parse_wiki_cards.py first.")
        sys.exit(1)

    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    # Find cards missing text
    missing = [c for c in cards if not c.get("text")]
    has_text = [c for c in cards if c.get("text")]
    print(f"\n  Total cards: {len(cards)}")
    print(f"  With text:   {len(has_text)}")
    print(f"  Missing text: {len(missing)}")

    if not missing:
        print("\n  All cards have text. Nothing to do.")
        return

    if args.limit:
        missing = missing[:args.limit]
        print(f"  Processing first {len(missing)} (--limit {args.limit})")

    # Dry run: just show which cards need text
    if args.dry_run:
        print(f"\n  Cards needing text extraction:")
        by_exp = {}
        for card in missing:
            exp = card["expansion"]
            by_exp.setdefault(exp, []).append(card["name"])

        for exp in sorted(by_exp.keys()):
            names = by_exp[exp]
            print(f"\n  {exp} ({len(names)} cards):")
            for name in names[:10]:
                print(f"    {name}")
            if len(names) > 10:
                print(f"    ... and {len(names) - 10} more")
        return

    # Connect
    print("\n  Connecting to OpenAI + Supabase...")
    openai_client, supabase = init_clients()
    print("  Connected.")

    # Process each card
    found = 0
    not_found = 0
    errors = 0

    # Build a lookup for fast card updating
    card_by_name = {c["name"]: c for c in cards}

    print(f"\n  Extracting text for {len(missing)} cards...\n")

    for i, card in enumerate(missing):
        name = card["name"]
        expansion = card["expansion"]
        print(f"  [{i+1}/{len(missing)}] {name} ({expansion})...", end=" ")

        # Find chunks mentioning this card
        chunks = find_chunks_mentioning_card(supabase, openai_client, name, expansion)

        if not chunks:
            print("no chunks found")
            not_found += 1
            time.sleep(0.2)
            continue

        # Extract card text via LLM
        text = extract_card_text(openai_client, name, chunks)

        if text:
            # Update the card
            attrs = parse_attributes(text)
            card_by_name[name]["text"] = text
            card_by_name[name].update(attrs)
            print(f"✅ \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
            found += 1
        else:
            print("❌ not found in chunks")
            not_found += 1

        # Rate limiting
        time.sleep(0.3)

    # Summary
    print(f"\n── Summary ───────────────────────────────")
    print(f"  Extracted:  {found}")
    print(f"  Not found:  {not_found}")
    print(f"  Errors:     {errors}")

    # Save updated cards
    updated_cards = list(card_by_name.values())
    with open(CARDS_JSON, "w", encoding="utf-8") as f:
        json.dump(updated_cards, f, indent=2, ensure_ascii=False)

    total_with_text = sum(1 for c in updated_cards if c.get("text"))
    print(f"\n  💾 Saved {len(updated_cards)} cards to {CARDS_JSON}")
    print(f"  Cards with text: {total_with_text}/{len(updated_cards)}")
    print(f"\n  Next step: python data/load_cards.py --validate")


if __name__ == "__main__":
    main()
