"""
scrape_strategy.py — Scrape Dominion Strategy Wiki articles (Playwright)

Uses Playwright (headless browser) to bypass the wiki's Anubis bot
protection and pull strategy articles for all cards via the MediaWiki API.

Cleans wikitext into plain text, chunks it, and optionally embeds + loads
into Supabase alongside the existing rulebook chunks.

Prerequisites:
  pip install playwright tiktoken
  playwright install chromium

Usage:
  python data/scrape_strategy.py --limit 5         # test with 5 cards
  python data/scrape_strategy.py                    # scrape all cards
  python data/scrape_strategy.py --embed            # scrape + embed + load
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Missing dependency: pip install playwright && playwright install chromium")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("Missing dependency: pip install tiktoken")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

WIKI_API = "https://wiki.dominionstrategy.com/api.php"
CARDS_JSON = Path("data/cards.json")
OUTPUT_JSON = Path("data/strategy_chunks.json")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
PAGE_LOAD_WAIT = 3000     # ms to wait for Anubis challenge
RATE_LIMIT_DELAY = 1.0    # seconds between requests (be polite)

enc = tiktoken.encoding_for_model("text-embedding-3-small")


# ─────────────────────────────────────────────────────────────────
# Wikitext cleaning
# ─────────────────────────────────────────────────────────────────

def clean_wikitext(wikitext: str) -> str:
    """Convert raw MediaWiki markup into readable plain text."""
    text = wikitext

    # Remove categories
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text)

    # Remove file/image links
    text = re.sub(r"\[\[File:[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[Image:[^\]]*\]\]", "", text)

    # Convert wiki links [[Page|Display]] → Display, [[Page]] → Page
    text = re.sub(r"\[\[[^\]]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)

    # Remove external links [url text] → text
    text = re.sub(r"\[https?://[^\s\]]+ ([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]*\]", "", text)

    # Remove templates {{...}} (nested, multiple passes)
    for _ in range(5):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Remove HTML tags
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    text = re.sub(r"<[^>]+>", "", text)

    # Remove table markup
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Convert headers === text === → text
    text = re.sub(r"={2,}\s*([^=]+?)\s*={2,}", r"\n\1\n", text)

    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", "", text)

    # Remove bullet/numbered list markers
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.MULTILINE)

    # Clean HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def extract_strategy_content(wikitext: str) -> str:
    """
    Extract strategy-relevant content from wikitext.
    Skips the infobox, trivia, secret history, gallery, etc.
    """
    # Remove the infobox template at the top
    text = re.sub(r"\{\{Infobox Card.*?\}\}", "", wikitext, flags=re.DOTALL)

    # Remove known non-strategy sections
    skip_patterns = [
        r"==\s*Trivia\s*==.*?(?===|\Z)",
        r"==\s*Versions?\s*==.*?(?===|\Z)",
        r"==\s*Secret History\s*==.*?(?===|\Z)",
        r"==\s*Gallery\s*==.*?(?===|\Z)",
        r"==\s*External [Ll]inks?\s*==.*?(?===|\Z)",
        r"==\s*In other languages?\s*==.*?(?===|\Z)",
        r"==\s*Other language versions\s*==.*?(?===|\Z)",
        r"==\s*Errata and Rulings\s*==.*?(?===|\Z)",
    ]
    for pattern in skip_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    return clean_wikitext(text)


# ─────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, card_name: str, expansion: str) -> list[dict]:
    """Chunk strategy text into overlapping token-based chunks."""
    tokens = enc.encode(text)

    if len(tokens) < 30:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()

        if len(chunk_text) > 30:
            chunks.append({
                "card_name": card_name,
                "expansion": expansion,
                "source_type": "strategy_wiki",
                "chunk_text": chunk_text,
            })

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ─────────────────────────────────────────────────────────────────
# Playwright scraper
# ─────────────────────────────────────────────────────────────────

def scrape_cards(card_list: list[dict], browser_page) -> tuple[list[dict], int, int]:
    """
    Scrape strategy articles for a list of cards using an open
    Playwright browser page.
    """
    all_chunks = []
    found = 0
    not_found = 0

    for i, card in enumerate(card_list):
        name = card["name"]
        expansion = card["expansion"]
        wiki_title = name.replace(" ", "_")
        url = f"{WIKI_API}?action=parse&page={wiki_title}&prop=wikitext&format=json"

        print(f"  [{i+1}/{len(card_list)}] {name}...", end=" ", flush=True)

        try:
            browser_page.goto(url, wait_until="networkidle", timeout=15000)
            browser_page.wait_for_timeout(PAGE_LOAD_WAIT)

            content = browser_page.inner_text("body")

            # Try parsing JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Might still be on Anubis page — wait and retry
                browser_page.wait_for_timeout(5000)
                content = browser_page.inner_text("body")
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    print("❌ bot check failed")
                    not_found += 1
                    time.sleep(RATE_LIMIT_DELAY)
                    continue

            if "error" in data:
                print("❌ no page")
                not_found += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
            if not wikitext or len(wikitext) < 50:
                print("⚠️ too short")
                not_found += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            strategy_text = extract_strategy_content(wikitext)

            if len(strategy_text) < 50:
                print(f"⚠️ no strategy content ({len(strategy_text)} chars)")
                not_found += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            chunks = chunk_text(strategy_text, name, expansion)
            all_chunks.extend(chunks)
            found += 1
            print(f"✅ {len(chunks)} chunks ({len(strategy_text)} chars)")

        except Exception as e:
            print(f"❌ error: {str(e)[:60]}")
            not_found += 1

        time.sleep(RATE_LIMIT_DELAY)

    return all_chunks, found, not_found


# ─────────────────────────────────────────────────────────────────
# Embedding + loading
# ─────────────────────────────────────────────────────────────────

def embed_and_load_strategy(chunks: list[dict]) -> None:
    """Embed strategy chunks and load into the rulebook_chunks table."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from openai import OpenAI
    from supabase import create_client

    openai_client = OpenAI()
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    print(f"\n  Embedding {len(chunks)} strategy chunks...")

    batch_size = 100
    all_records = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["chunk_text"] for c in batch]

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]

        for chunk, embedding in zip(batch, embeddings):
            all_records.append({
                "expansion": chunk["expansion"],
                "source_page": 0,
                "chunk_text": f"[Strategy: {chunk['card_name']}] {chunk['chunk_text']}",
                "embedding": embedding,
            })

        print(f"    Embedded batch [{i+1}–{i+len(batch)}]")
        time.sleep(0.2)

    # Clear existing strategy chunks
    print("  Clearing existing strategy chunks...")
    try:
        supabase.table("rulebook_chunks") \
            .delete() \
            .like("chunk_text", "[Strategy:%") \
            .execute()
    except Exception:
        pass

    # Insert in batches
    print(f"  Loading {len(all_records)} chunks into Supabase...")
    loaded = 0
    for i in range(0, len(all_records), 50):
        batch = all_records[i:i + 50]
        try:
            supabase.table("rulebook_chunks").insert(batch).execute()
            loaded += len(batch)
            print(f"    Inserted [{i+1}–{i+len(batch)}]")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    print(f"\n  ✅ Loaded {loaded} strategy chunks into rulebook_chunks table.")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape Dominion strategy articles.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many cards.")
    parser.add_argument("--embed", action="store_true",
                        help="Also embed and load into Supabase.")
    args = parser.parse_args()

    print("=" * 50)
    print("DominionSage — Strategy Scraper (Playwright)")
    print("=" * 50)

    # Load card list
    if not CARDS_JSON.exists():
        print(f"Error: {CARDS_JSON} not found.")
        sys.exit(1)

    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    # Deduplicate and skip basic supply cards
    skip_names = {"Copper", "Silver", "Gold", "Platinum", "Estate", "Duchy",
                  "Province", "Colony", "Curse", "Potion"}
    card_list = []
    seen = set()
    for card in cards:
        name = card["name"]
        if name not in seen and name not in skip_names:
            card_list.append(card)
            seen.add(name)

    if args.limit:
        card_list = card_list[:args.limit]

    print(f"  Cards to process: {len(card_list)}")
    print(f"  Estimated time: ~{len(card_list) * 1.5 / 60:.0f} minutes")

    # Launch browser
    print(f"\n  Launching browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Warm up — pass the Anubis challenge once, cookie persists
        print("  Warming up (passing bot check)...")
        page.goto(f"{WIKI_API}?action=query&meta=siteinfo&format=json",
                  wait_until="networkidle", timeout=20000)
        page.wait_for_timeout(5000)

        warmup = page.inner_text("body")
        if "Making sure" in warmup:
            print("  ⚠️ Still on bot check. Waiting longer...")
            page.wait_for_timeout(10000)
            warmup = page.inner_text("body")

        if "mediawiki" in warmup.lower():
            print("  ✅ Bot check passed. Starting scrape.\n")
        else:
            print("  ⚠️ Might not have passed bot check. Trying anyway.\n")

        # Scrape
        all_chunks, found, not_found = scrape_cards(card_list, page)

        browser.close()

    # Summary
    print(f"\n── Summary ───────────────────────────────")
    print(f"  Pages found:     {found}")
    print(f"  Pages not found: {not_found}")
    print(f"  Total chunks:    {len(all_chunks)}")

    if all_chunks:
        total_tokens = sum(len(enc.encode(c["chunk_text"])) for c in all_chunks)
        est_cost = (total_tokens / 1_000_000) * 0.02
        print(f"  Total tokens:    {total_tokens:,}")
        print(f"  Est. embed cost: ${est_cost:.4f}")

    # Save chunks
    if all_chunks:
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 Saved {len(all_chunks)} chunks to {OUTPUT_JSON}")

    if args.embed and all_chunks:
        embed_and_load_strategy(all_chunks)
    elif all_chunks:
        print(f"\n  Run again with --embed to embed and load into Supabase.")


if __name__ == "__main__":
    main()
