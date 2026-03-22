"""
rules_search.py — DominionSage Vector Search (Phase 3)

Handles Type 3 (rules question) queries by embedding the question
and finding the most similar rulebook chunks via pgvector cosine
similarity.

This is the "vector path" of the hybrid retrieval system. It handles
open-ended questions where you don't know which column to query —
instead, you search by meaning.

How it works:
  1. The user's question is converted to a 1,536-dimension vector
  2. That vector is compared to every chunk's vector using cosine similarity
  3. The top K most similar chunks are returned as context

Analogy: Traditional SQL search is like looking up a word in the index
of a textbook — you need the exact word. Vector search is like asking
a librarian "What section covers timing rules for instant-speed effects?"
The librarian understands the meaning and points you to the right shelf,
even if the book never uses the phrase "instant-speed."
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

try:
    from supabase import create_client, Client
except ImportError:
    print("Missing dependency: pip install supabase")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Client initialization
# ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"

_openai: OpenAI | None = None
_supabase: Client | None = None


def _get_clients() -> tuple[OpenAI, Client]:
    """Lazy-initialize both clients."""
    global _openai, _supabase

    if _openai is None or _supabase is None:
        if load_dotenv:
            load_dotenv()

        _openai = OpenAI()
        _supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY"),
        )

    return _openai, _supabase


# ─────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Convert a text query into a vector embedding.

    This costs ~$0.00002 per query (negligible), but it does add
    ~200ms of latency for the API call. In production, you'd cache
    frequent queries.
    """
    openai_client, _ = _get_clients()

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────

def search_rules(
    query: str,
    top_k: int = 3,
    expansion: str | None = None,
    min_similarity: float = 0.25,
) -> list[dict]:
    """
    Semantic search over rulebook chunks.

    Args:
        query:          The user's natural language question.
        top_k:          Number of results to return (default 3).
        expansion:      If set, only search chunks from this expansion.
        min_similarity:  Drop chunks below this similarity threshold.
                        This prevents low-relevance chunks from being
                        cited in the answer. 0.25 is a conservative
                        threshold — increase if you see irrelevant
                        sources, decrease if you're missing results.

    Returns:
        List of dicts with: expansion, source_page, chunk_text, similarity
    """
    _, supabase = _get_clients()

    # Step 1: embed the query
    query_embedding = embed_query(query)

    # Step 2: call the appropriate match function
    # Request more than top_k so we have room after filtering
    fetch_count = top_k * 2

    try:
        if expansion:
            result = supabase.rpc("match_chunks_by_expansion", {
                "query_embedding": query_embedding,
                "match_count": fetch_count,
                "filter_expansion": expansion,
            }).execute()
        else:
            result = supabase.rpc("match_chunks", {
                "query_embedding": query_embedding,
                "match_count": fetch_count,
            }).execute()

        if not result.data:
            return []

        # Step 3: filter by confidence threshold
        filtered = [
            chunk for chunk in result.data
            if chunk.get("similarity", 0) >= min_similarity
        ]

        # Return at most top_k results
        return filtered[:top_k]

    except Exception as e:
        print(f"  ⚠️  Rules search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "When can I play Reaction cards?",
        "How do Duration cards work?",
        "What happens when the supply runs out?",
    ]

    print("── Rules Search Tests ────────────────────")
    for query in test_queries:
        print(f"\n  Q: \"{query}\"")
        results = search_rules(query, top_k=3)
        if results:
            for r in results:
                sim = f"{r.get('similarity', 0):.3f}"
                exp = r.get("expansion", "?")
                page = r.get("source_page", "?")
                text = r.get("chunk_text", "")[:100]
                print(f"    [{sim}] {exp} p.{page}: {text}...")
        else:
            print("    No results.")
