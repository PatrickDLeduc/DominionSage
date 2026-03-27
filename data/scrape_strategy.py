"""
scrape_strategy.py — Scrape Dominion Strategy Wiki articles (Playwright)

Uses Playwright (headless browser) to bypass the wiki's Anubis bot
protection and pull strategy articles for all cards via the MediaWiki API.

Cleans wikitext into plain text, splits into semantic chunks (one chunk
per card overview, synergy, opening principle, or archetype observation),
and optionally embeds + loads into Supabase.

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
MAX_CHUNK_TOKENS = 1500   # max tokens per semantic chunk (well under 8191 embedding limit)
FALLBACK_OVERLAP = 100    # overlap for sliding window fallback on oversized paragraphs
PAGE_LOAD_WAIT = 3000     # ms to wait for Anubis challenge
RATE_LIMIT_DELAY = 1.0    # seconds between requests (be polite)

enc = tiktoken.encoding_for_model("text-embedding-3-small")

# Card name set for related_cards extraction (loaded lazily)
_ALL_CARD_NAMES: set[str] | None = None


def _load_card_names() -> set[str]:
    """Load all card names from cards.json for related_cards extraction."""
    global _ALL_CARD_NAMES
    if _ALL_CARD_NAMES is not None:
        return _ALL_CARD_NAMES
    if CARDS_JSON.exists():
        with open(CARDS_JSON, "r", encoding="utf-8") as f:
            cards = json.load(f)
        _ALL_CARD_NAMES = {c["name"] for c in cards}
    else:
        _ALL_CARD_NAMES = set()
    return _ALL_CARD_NAMES


# ─────────────────────────────────────────────────────────────────
# Wikitext cleaning
# ─────────────────────────────────────────────────────────────────

def clean_wikitext(wikitext: str) -> str:
    """Convert raw MediaWiki markup into readable plain text.

    NOTE: This does NOT convert == headers ==. Headers are handled
    separately by split_into_sections() before this function is called.
    """
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

    # Strip any remaining == headers == (subsection headers within a section body)
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


# ─────────────────────────────────────────────────────────────────
# Section splitting & semantic chunking
# ─────────────────────────────────────────────────────────────────

# Sections to skip entirely (non-strategy content)
_SKIP_SECTIONS = {
    "trivia", "versions", "version", "secret history", "gallery",
    "external links", "external link", "in other languages",
    "other language versions", "errata and rulings",
    "english versions", "preview", "relevant outtakes",
    "donald x.'s opinion", "donald x's opinion", "retrospective",
    "other strategy articles", "previous versions",
}

# Keywords that signal an "opening" paragraph
_OPENING_KEYWORDS = re.compile(
    r"\b(opening|open with|first buy|turn [12]\b|3/4 split|5/2 split|4/3 split|"
    r"2/5 split|opener|open[s]?\b.*\bwith\b)",
    re.IGNORECASE,
)

# Keywords that signal a "kingdom archetype" paragraph
_ARCHETYPE_KEYWORDS = re.compile(
    r"\b(engine|big money|rush|slog|archetype|this kingdom|kingdom.*type|"
    r"money strategy|combo deck|alt[- ]?vp)",
    re.IGNORECASE,
)


def _strip_non_strategy_sections(wikitext: str) -> str:
    """Remove infobox and known non-strategy sections from raw wikitext."""
    text = re.sub(r"\{\{Infobox Card.*?\}\}", "", wikitext, flags=re.DOTALL)

    # Remove non-strategy sections (match ==Section== through to next == or end)
    for section_name in _SKIP_SECTIONS:
        pattern = rf"==\s*{re.escape(section_name)}\s*==.*?(?===|\Z)"
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    return text


def split_into_sections(wikitext: str) -> list[tuple[str, str]]:
    """Split raw wikitext into (section_name, raw_body) pairs.

    Operates on raw wikitext BEFORE clean_wikitext is called, so that
    == header == boundaries are preserved for semantic splitting.
    """
    text = _strip_non_strategy_sections(wikitext)

    # Split on top-level == headers == (captures the header itself)
    parts = re.split(r"(={2}\s*[^=]+?\s*={2})", text)

    sections = []
    # Content before the first header is the "intro"
    intro = parts[0].strip()
    if intro:
        sections.append(("intro", intro))

    # Pair each header with its body
    i = 1
    while i < len(parts) - 1:
        header_raw = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # Extract header name from == Name ==
        header_match = re.match(r"={2}\s*(.+?)\s*={2}", header_raw)
        header_name = header_match.group(1).strip() if header_match else "unknown"
        sections.append((header_name, body.strip()))
        i += 2

    return sections


def _classify_section(section_name: str) -> str | None:
    """Map a section header name to a chunk_type, or None to skip."""
    name_lower = section_name.lower().strip()

    if name_lower in _SKIP_SECTIONS:
        return None

    if name_lower in ("faq", "official faq", "other rules clarifications"):
        return "faq"
    if "synerg" in name_lower or "combo" in name_lower:
        return "synergy"
    if name_lower in ("antisynergies", "anti-synergies"):
        return "synergy"

    # Strategy, intro, and anything else that survived filtering
    return "overview"


def _classify_paragraph(text: str) -> str:
    """Classify a strategy paragraph by keyword signals."""
    if _OPENING_KEYWORDS.search(text):
        return "opening"
    if _ARCHETYPE_KEYWORDS.search(text):
        return "archetype"
    return "overview"


def _find_related_cards(text: str, card_name: str) -> list[str]:
    """Find card names mentioned in a chunk of text."""
    card_names = _load_card_names()
    related = []
    for name in card_names:
        if name == card_name:
            continue
        # Match whole word (avoid partial matches like "Inn" in "Inning")
        if re.search(r'\b' + re.escape(name) + r'\b', text):
            related.append(name)
    return sorted(related)


def _chunk_type_prefix(chunk_type: str, card_name: str) -> str:
    """Generate the semantic prefix for a chunk."""
    prefixes = {
        "overview": f"[Strategy Overview: {card_name}]",
        "synergy": f"[Synergy: {card_name}]",
        "opening": f"[Opening: {card_name}]",
        "archetype": f"[Archetype: {card_name}]",
        "faq": f"[FAQ: {card_name}]",
    }
    return prefixes.get(chunk_type, f"[Strategy: {card_name}]")


def _sliding_window_fallback(text: str, card_name: str, expansion: str,
                              chunk_type: str) -> list[dict]:
    """Fall back to sliding window for a single oversized paragraph."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + MAX_CHUNK_TOKENS
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens).strip()
        if len(chunk_text_str) > 30:
            prefix = _chunk_type_prefix(chunk_type, card_name)
            chunks.append({
                "card_name": card_name,
                "expansion": expansion,
                "source_type": "strategy_wiki",
                "chunk_type": chunk_type,
                "chunk_text": f"{prefix} {chunk_text_str}",
                "related_cards": _find_related_cards(chunk_text_str, card_name),
            })
        start += MAX_CHUNK_TOKENS - FALLBACK_OVERLAP
    return chunks


def _split_synergy_items(raw_body: str) -> list[str]:
    """Split a synergy section into individual synergy items.

    Wiki synergy sections typically use list items (* or #) or
    blank-line-separated paragraphs.
    """
    # Split on list item markers in raw wikitext
    items = re.split(r"\n(?=[*#])", raw_body)

    # If no list items found, split on blank lines
    if len(items) <= 1:
        items = re.split(r"\n\s*\n", raw_body)

    return [item.strip() for item in items if item.strip()]


def semantic_chunk(wikitext: str, card_name: str, expansion: str) -> list[dict]:
    """Split a wiki article into semantic chunks.

    Each chunk is one semantic unit: a card overview paragraph, an
    individual synergy, an opening principle, or an archetype observation.
    """
    sections = split_into_sections(wikitext)
    chunks = []

    for section_name, raw_body in sections:
        chunk_type = _classify_section(section_name)
        if chunk_type is None:
            continue

        if chunk_type == "synergy":
            # Split into individual synergy items
            items = _split_synergy_items(raw_body)
            for item in items:
                cleaned = clean_wikitext(item)
                if len(cleaned) < 30:
                    continue
                token_count = len(enc.encode(cleaned))
                if token_count > MAX_CHUNK_TOKENS:
                    chunks.extend(_sliding_window_fallback(
                        cleaned, card_name, expansion, "synergy"))
                else:
                    prefix = _chunk_type_prefix("synergy", card_name)
                    chunks.append({
                        "card_name": card_name,
                        "expansion": expansion,
                        "source_type": "strategy_wiki",
                        "chunk_type": "synergy",
                        "chunk_text": f"{prefix} {cleaned}",
                        "related_cards": _find_related_cards(cleaned, card_name),
                    })

        elif chunk_type == "faq":
            cleaned = clean_wikitext(raw_body)
            if len(cleaned) < 30:
                continue
            token_count = len(enc.encode(cleaned))
            if token_count > MAX_CHUNK_TOKENS:
                chunks.extend(_sliding_window_fallback(
                    cleaned, card_name, expansion, "faq"))
            else:
                prefix = _chunk_type_prefix("faq", card_name)
                chunks.append({
                    "card_name": card_name,
                    "expansion": expansion,
                    "source_type": "strategy_wiki",
                    "chunk_type": "faq",
                    "chunk_text": f"{prefix} {cleaned}",
                    "related_cards": _find_related_cards(cleaned, card_name),
                })

        else:
            # Overview / strategy — split by paragraph and classify each
            cleaned = clean_wikitext(raw_body)
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

            for para in paragraphs:
                if len(para) < 30:
                    continue
                para_type = _classify_paragraph(para)
                token_count = len(enc.encode(para))
                if token_count > MAX_CHUNK_TOKENS:
                    chunks.extend(_sliding_window_fallback(
                        para, card_name, expansion, para_type))
                else:
                    prefix = _chunk_type_prefix(para_type, card_name)
                    chunks.append({
                        "card_name": card_name,
                        "expansion": expansion,
                        "source_type": "strategy_wiki",
                        "chunk_type": para_type,
                        "chunk_text": f"{prefix} {para}",
                        "related_cards": _find_related_cards(para, card_name),
                    })

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

            chunks = semantic_chunk(wikitext, name, expansion)

            if not chunks:
                print(f"⚠️ no strategy content")
                not_found += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            all_chunks.extend(chunks)
            found += 1
            type_counts = {}
            for c in chunks:
                t = c.get("chunk_type", "?")
                type_counts[t] = type_counts.get(t, 0) + 1
            type_str = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items()))
            print(f"✅ {len(chunks)} chunks ({type_str})")

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
        # chunk_text already has the semantic prefix baked in
        texts = [c["chunk_text"] for c in batch]

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]

        for chunk, embedding in zip(batch, embeddings):
            record = {
                "expansion": chunk["expansion"],
                "source_page": 0,
                "chunk_text": chunk["chunk_text"],
                "embedding": embedding,
            }
            # Pass through new columns if the table supports them
            if "chunk_type" in chunk:
                record["chunk_type"] = chunk["chunk_type"]
            if "card_name" in chunk:
                record["card_name"] = chunk["card_name"]
            all_records.append(record)

        print(f"    Embedded batch [{i+1}–{i+len(batch)}]")
        time.sleep(0.2)

    # Clear existing strategy chunks (source_page=0 identifies strategy chunks)
    print("  Clearing existing strategy chunks...")
    try:
        supabase.table("rulebook_chunks") \
            .delete() \
            .eq("source_page", 0) \
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

        # Chunk type distribution
        type_counts = {}
        for c in all_chunks:
            t = c.get("chunk_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"\n  Chunk types:")
        for t, count in sorted(type_counts.items()):
            print(f"    {t:<12} {count:>5}")

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
