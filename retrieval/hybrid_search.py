"""
hybrid_search.py — DominionSage Hybrid Search with Reciprocal Rank Fusion

Combines vector search (semantic similarity) and BM25 (keyword matching)
using Reciprocal Rank Fusion (RRF) to get the best of both worlds.

Why RRF over other fusion methods?
  The challenge with combining vector and BM25 scores is that they're on
  completely different scales — cosine similarity ranges from 0 to 1,
  while BM25 scores can be any positive number. You can't just add them.

  RRF solves this elegantly: instead of combining scores, it combines
  RANKS. If a document is ranked #1 by both systems, it gets a high
  fused score. If it's #1 by one system but absent from the other,
  it still gets credit but less.

  The formula: rrf_score(d) = sum(1 / (k + rank_i(d))) for each system i.
  k=60 is the standard constant (from the original 2009 Cormack paper).

  Analogy: Imagine two music fans making their top-10 lists. Even if
  they use completely different rating criteria, just combining their
  rankings produces a good consensus list — songs both fans love will
  appear near the top, while niche picks will settle lower.
"""

from retrieval.rules_search import search_rules
from retrieval.bm25_search import bm25_search


# ─────────────────────────────────────────────────────────────────
# RRF Fusion
# ─────────────────────────────────────────────────────────────────

RRF_K = 60  # Standard RRF constant from Cormack et al. (2009)


def _normalize_text(text: str) -> str:
    """Normalize chunk text for deduplication."""
    return text.strip().lower()


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Fuse two ranked result lists using Reciprocal Rank Fusion.

    Each result gets a score of 1/(k + rank) from each list it appears in.
    Results appearing in both lists get the sum of both scores.

    Args:
        vector_results: Results from vector/embedding search (must have 'chunk_text').
        bm25_results:   Results from BM25 keyword search (must have 'chunk_text').
        top_k:          Number of fused results to return.

    Returns:
        List of result dicts sorted by fused RRF score, with 'rrf_score' added.
    """
    # Build a map from normalized chunk text -> best result dict + RRF score
    fused: dict[str, dict] = {}

    # Score vector results by rank
    for rank, result in enumerate(vector_results, start=1):
        key = _normalize_text(result.get("chunk_text", ""))
        if not key:
            continue

        rrf_score = 1.0 / (RRF_K + rank)
        
        # Boost core rules (front of book) over card appendices (back of book)
        try:
            page = str(result.get("source_page", ""))
            if page.isdigit() and int(page) <= 8:
                rrf_score *= 1.5
        except Exception:
            pass

        if key not in fused:
            fused[key] = {
                **result,
                "rrf_score": rrf_score,
                "in_vector": True,
                "in_bm25": False,
            }
        else:
            fused[key]["rrf_score"] += rrf_score
            fused[key]["in_vector"] = True

    # Score BM25 results by rank
    for rank, result in enumerate(bm25_results, start=1):
        key = _normalize_text(result.get("chunk_text", ""))
        if not key:
            continue

        rrf_score = 1.0 / (RRF_K + rank)
        
        # Boost core rules (front of book) over card appendices (back of book)
        try:
            page = str(result.get("source_page", ""))
            if page.isdigit() and int(page) <= 8:
                rrf_score *= 1.5
        except Exception:
            pass

        if key not in fused:
            fused[key] = {
                **result,
                "rrf_score": rrf_score,
                "in_vector": False,
                "in_bm25": True,
            }
        else:
            fused[key]["rrf_score"] += rrf_score
            fused[key]["in_bm25"] = True

    # Sort by fused score and return top_k
    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────
# Main hybrid search entry point
# ─────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = 5,
    expansion: str | None = None,
) -> list[dict]:
    """
    Hybrid search combining vector similarity and BM25 keyword matching.

    This is the drop-in replacement for search_rules() in the orchestrator.
    Returns the same format so downstream code (synthesizer, UI) needs
    no changes.

    Args:
        query:     The user's natural language question.
        top_k:     Number of results to return.
        expansion: Optional expansion filter.

    Returns:
        List of dicts with: expansion, source_page, chunk_text, similarity/bm25_score,
        plus rrf_score for debugging.
    """
    # Fetch more candidates from each system than we need, so RRF has
    # plenty to work with. We'll trim to top_k after fusion.
    fetch_k = top_k * 2

    # Run both searches
    vector_results = search_rules(query, top_k=fetch_k, expansion=expansion)
    bm25_results = bm25_search(query, top_k=fetch_k, expansion=expansion)

    # Fuse with RRF
    fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)

    return fused


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "Duration cards",
        "When can I play Reaction cards?",
        "What happens when the supply runs out?",
    ]

    print("── Hybrid Search Comparison ──────────────")

    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"  Q: \"{query}\"")

        # Vector only
        vec = search_rules(query, top_k=3)
        print(f"\n  Vector (top 3):")
        for r in vec:
            sim = f"{r.get('similarity', 0):.3f}"
            text = r.get("chunk_text", "")[:80]
            print(f"    [sim {sim}] {text}...")

        # BM25 only
        bm = bm25_search(query, top_k=3)
        print(f"\n  BM25 (top 3):")
        for r in bm:
            score = f"{r.get('bm25_score', 0):.3f}"
            text = r.get("chunk_text", "")[:80]
            print(f"    [bm25 {score}] {text}...")

        # Hybrid (RRF)
        hyb = hybrid_search(query, top_k=3)
        print(f"\n  Hybrid RRF (top 3):")
        for r in hyb:
            rrf = f"{r.get('rrf_score', 0):.4f}"
            v = "✓" if r.get("in_vector") else "✗"
            b = "✓" if r.get("in_bm25") else "✗"
            text = r.get("chunk_text", "")[:80]
            print(f"    [rrf {rrf}] vec:{v} bm25:{b} {text}...")
