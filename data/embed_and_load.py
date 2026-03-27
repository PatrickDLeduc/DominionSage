"""
embed_and_load.py — DominionSage Embedding Loader (Phase 2)

Reads data/rulebook_chunks.json (output of chunk_rulebooks.py), generates
vector embeddings via OpenAI, and loads everything into the Supabase
`rulebook_chunks` table.

How embeddings work (the intuition):
  An embedding turns text into a list of 1,536 numbers. Texts that mean
  similar things end up close together in this 1,536-dimensional space,
  even if they use completely different words. "When can I play Reaction
  cards?" and "Timing rules for revealing Reactions" would have high
  cosine similarity despite sharing almost no words.

  Think of it like GPS coordinates for meaning — two restaurants in the
  same neighborhood have similar coordinates, even if they have totally
  different names.

Prerequisites:
  pip install openai supabase python-dotenv

Usage:
  python data/embed_and_load.py                 # embed + load all chunks
  python data/embed_and_load.py --dry-run       # show what would be loaded
  python data/embed_and_load.py --batch-size 50 # adjust batch size
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

CHUNKS_JSON = Path("data/rulebook_chunks.json")
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, $0.02/1M tokens
BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per batch call
RATE_LIMIT_DELAY = 0.1  # seconds between batches (gentle rate limiting)


def init_clients() -> tuple[OpenAI, Client]:
    """Initialize OpenAI and Supabase clients."""
    if load_dotenv:
        load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    supa_url = os.getenv("SUPABASE_URL")
    supa_key = os.getenv("SUPABASE_KEY")

    missing = []
    if not openai_key: missing.append("OPENAI_API_KEY")
    if not supa_url:   missing.append("SUPABASE_URL")
    if not supa_key:   missing.append("SUPABASE_KEY")

    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Set them in your .env file or export them.")
        sys.exit(1)

    return OpenAI(), create_client(supa_url, supa_key)


# ─────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────

def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Uses the batch endpoint (multiple texts per API call) instead of
    one-at-a-time. This is faster and slightly cheaper due to fewer
    HTTP round-trips.

    Analogy: Instead of mailing 100 letters individually, you put
    them all in one box and ship them together.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # Return embeddings in the same order as the input texts
    return [item.embedding for item in response.data]


# ─────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────

def clear_existing_chunks(supabase: Client, expansion: str) -> int:
    """
    Delete existing chunks for an expansion before re-loading.
    This makes the script idempotent — safe to run multiple times.
    Returns the count of deleted rows.
    """
    try:
        result = supabase.table("rulebook_chunks") \
            .delete() \
            .eq("expansion", expansion) \
            .execute()
        return len(result.data) if result.data else 0
    except Exception:
        return 0


def load_chunks_to_supabase(
    supabase: Client,
    chunks: list[dict],
    batch_size: int = 50,
) -> dict:
    """
    Insert chunks with embeddings into Supabase.
    Returns a summary dict.
    """
    summary = {"total": len(chunks), "success": 0, "errors": []}

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_label = f"[{i + 1}–{i + len(batch)}]"

        try:
            supabase.table("rulebook_chunks").insert(batch).execute()
            summary["success"] += len(batch)
            print(f"    Inserted batch {batch_label}")
        except Exception as e:
            summary["errors"].append({"batch": batch_label, "error": str(e)})
            print(f"    ❌ Error on batch {batch_label}: {e}")

    return summary


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_with_test_query(openai_client: OpenAI, supabase: Client) -> None:
    """
    Run a quick semantic search to verify everything is working.
    This is the Phase 2.4 validation from the walkthrough.
    """
    test_queries = [
        "When can I play Reaction cards?",
        "How do Duration cards work?",
        "What happens when the Supply runs out?",
    ]

    print("\n── Validation: Test Queries ───────────────")

    for query in test_queries:
        print(f"\n  Q: \"{query}\"")

        # Embed the query
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = response.data[0].embedding

        # Search via pgvector (using Supabase RPC or direct query)
        # This uses a raw RPC call to the match_chunks function.
        # If you haven't created it yet, we fall back to a direct approach.
        try:
            result = supabase.rpc("match_chunks", {
                "query_embedding": query_embedding,
                "match_count": 3,
            }).execute()
            results = result.data
        except Exception:
            # Fallback: use the PostgREST ordering (less elegant but works)
            # This won't work via the client directly for cosine distance,
            # so we'll just verify the data exists.
            result = supabase.table("rulebook_chunks") \
                .select("expansion, source_page, chunk_text") \
                .limit(3) \
                .execute()
            results = result.data
            print("    (Note: match_chunks RPC not found — showing sample rows instead)")
            print("    Create the function with the SQL in the walkthrough Phase 2.4)")

        if results:
            for r in results[:3]:
                expansion = r.get("expansion", "?")
                page = r.get("source_page", "?")
                text_preview = r.get("chunk_text", "")[:80]
                similarity = r.get("similarity", "n/a")
                print(f"    → [{expansion} p.{page}] (sim: {similarity}) {text_preview}...")
        else:
            print("    ⚠️  No results returned. Check your table and RPC function.")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Embed and load rulebook chunks into Supabase."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be embedded without calling APIs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Texts per embedding API call (default: {BATCH_SIZE})")
    parser.add_argument("--skip-clear", action="store_true",
                        help="Don't delete existing chunks before loading.")
    parser.add_argument("--validate", action="store_true",
                        help="Run test queries after loading.")
    args = parser.parse_args()

    print("=" * 50)
    print("DominionSage — Embedding Loader")
    print("=" * 50)

    # Load chunks JSON
    if not CHUNKS_JSON.exists():
        print(f"Error: {CHUNKS_JSON} not found.")
        print("Run `python data/chunk_rulebooks.py` first.")
        sys.exit(1)

    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"  Loaded {len(chunks)} chunks from {CHUNKS_JSON}")

    # Group by expansion for reporting
    by_expansion = {}
    for chunk in chunks:
        exp = chunk["expansion"]
        by_expansion.setdefault(exp, []).append(chunk)

    print(f"  Expansions: {', '.join(sorted(by_expansion.keys()))}")

    # ── Dry run ──
    if args.dry_run:
        print("\n── Dry Run (no API calls) ─────────────────")
        for exp, exp_chunks in sorted(by_expansion.items()):
            print(f"  {exp}: {len(exp_chunks)} chunks")
            # Show first chunk as sample
            sample = exp_chunks[0]["chunk_text"][:100]
            print(f"    Sample: \"{sample}...\"")

        # Estimate cost
        total_chars = sum(len(c["chunk_text"]) for c in chunks)
        est_tokens = total_chars / 4  # rough estimate: 1 token ≈ 4 chars
        est_cost = (est_tokens / 1_000_000) * 0.02
        print(f"\n  Estimated total tokens: ~{int(est_tokens):,}")
        print(f"  Estimated cost: ~${est_cost:.4f}")
        return

    # ── Connect ──
    print("\nConnecting to OpenAI + Supabase...")
    openai_client, supabase = init_clients()
    print("  Connected.")

    # ── Process each expansion ──
    total_cost_tokens = 0
    total_loaded = 0

    for exp, exp_chunks in sorted(by_expansion.items()):
        print(f"\n── {exp} ({len(exp_chunks)} chunks) ──")

        # Clear existing data for this expansion (idempotent)
        if not args.skip_clear:
            deleted = clear_existing_chunks(supabase, exp)
            if deleted > 0:
                print(f"  Cleared {deleted} existing chunks for '{exp}'.")

        # Embed in batches
        print(f"  Embedding...")
        texts = [c["chunk_text"] for c in exp_chunks]
        all_embeddings = []

        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i : i + args.batch_size]
            batch_label = f"{i + 1}–{i + len(batch_texts)}"

            embeddings = embed_batch(openai_client, batch_texts)
            all_embeddings.extend(embeddings)
            print(f"    Embedded batch [{batch_label}] ({len(batch_texts)} chunks)")

            # Track token usage (approximate: re-count is cheap)
            total_cost_tokens += sum(len(t) for t in batch_texts) // 4

            time.sleep(RATE_LIMIT_DELAY)

        # Prepare records for Supabase
        records = []
        for chunk, embedding in zip(exp_chunks, all_embeddings):
            record = {
                "expansion": chunk["expansion"],
                "source_page": chunk["source_page"],
                "chunk_text": chunk["chunk_text"],
                "embedding": embedding,
            }
            # Pass through semantic metadata if present (strategy chunks)
            if "chunk_type" in chunk:
                record["chunk_type"] = chunk["chunk_type"]
            if "card_name" in chunk:
                record["card_name"] = chunk["card_name"]
            records.append(record)

        # Load into Supabase
        print(f"  Loading into Supabase...")
        summary = load_chunks_to_supabase(supabase, records)
        total_loaded += summary["success"]

        if summary["errors"]:
            for err in summary["errors"]:
                print(f"    ❌ {err['batch']}: {err['error']}")

    # ── Final summary ──
    est_cost = (total_cost_tokens / 1_000_000) * 0.02
    print(f"\n── Final Summary ─────────────────────────")
    print(f"  Chunks embedded:  {len(chunks)}")
    print(f"  Chunks loaded:    {total_loaded}")
    print(f"  Est. tokens used: ~{total_cost_tokens:,}")
    print(f"  Est. cost:        ~${est_cost:.4f}")

    if total_loaded == len(chunks):
        print(f"\n  ✅ All {total_loaded} chunks loaded successfully!")
    else:
        print(f"\n  ⚠️  {len(chunks) - total_loaded} chunks failed to load.")

    # ── Validation ──
    if args.validate:
        validate_with_test_query(openai_client, supabase)
    else:
        print("\n  Tip: run with --validate to test semantic search.")



if __name__ == "__main__":
    main()
