"""
bm25_search.py — DominionSage BM25 Keyword Search

Provides local keyword-based search using BM25 (Best Matching 25) over
rulebook and strategy chunks. This complements the vector search in
rules_search.py — vector search finds semantically similar text, while
BM25 finds text with matching keywords.

Why BM25 alongside vectors?
  Vector search excels at meaning: "What happens at end of game?" matches
  "supply pile depletion" even though they share no words. But it can miss
  exact keyword hits: a query for "Duration cards" might not rank a chunk
  containing that exact phrase as highly as one that's semantically similar
  but uses different wording.

  BM25 is the opposite — it scores documents by how often the query's
  exact words appear (with TF-IDF weighting). "Duration cards" will
  strongly match chunks containing those exact words.

  Together, they cover each other's blind spots.

How BM25 works (the intuition):
  BM25 scores each document by:
    1. How often each query term appears in the document (term frequency)
    2. How rare that term is across ALL documents (inverse document frequency)
    3. A length normalization so short docs aren't penalized

  Think of it like Google circa 2005 — before neural embeddings existed,
  keyword matching with smart weighting was state of the art. It's still
  competitive for exact-match queries.

Architecture:
  The BM25 index is built once from the JSON chunk files (rulebook_chunks.json
  and strategy_chunks.json) and kept in memory as a singleton. At ~3MB of
  text, this is negligible memory. Queries take ~5ms with no API calls.
"""

import json
import re
import sys
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Missing dependency: pip install rank-bm25")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────

# Simple stopwords — common English words that add noise to BM25 scoring.
# Dominion-specific terms like "card", "action", "play" are intentionally
# NOT in this list because they carry meaning in this domain.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "and", "but", "or", "not",
    "no", "if", "then", "than", "that", "this", "it", "its", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "our", "their", "what", "which", "who", "when",
    "where", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "same", "so", "very",
}


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25: lowercase, map common numerals to words,
    split on non-alphanumeric, remove stopwords and very short tokens.

    We keep it simple because BM25 is robust to tokenization quality —
    it's the term frequency / inverse document frequency weighting
    that does the heavy lifting.
    """
    text = text.lower()
    # Map common numbers so query "3" matches rule text "three"
    text = text.replace("1", "one").replace("2", "two")\
               .replace("3", "three").replace("4", "four").replace("5", "five")
               
    # Split on non-word characters
    tokens = re.findall(r"[a-z0-9]+", text)
    # Remove stopwords and single-character tokens
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ─────────────────────────────────────────────────────────────────
# BM25 Index (singleton)
# ─────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_bm25_index: BM25Okapi | None = None
_corpus_chunks: list[dict] | None = None


def _load_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Build the BM25 index from JSON chunk files (lazy singleton).

    Loads both rulebook_chunks.json and strategy_chunks.json, tokenizes
    each chunk, and builds a BM25Okapi index. Called once on first query,
    then cached for the lifetime of the process.
    """
    global _bm25_index, _corpus_chunks

    if _bm25_index is not None and _corpus_chunks is not None:
        return _bm25_index, _corpus_chunks

    all_chunks = []

    # Load rulebook chunks
    rulebook_path = _DATA_DIR / "rulebook_chunks.json"
    if rulebook_path.exists():
        with open(rulebook_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for chunk in chunks:
                chunk["source_type"] = "rulebook"
            all_chunks.extend(chunks)

    # Load strategy chunks
    strategy_path = _DATA_DIR / "strategy_chunks.json"
    if strategy_path.exists():
        with open(strategy_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for chunk in chunks:
                chunk["source_type"] = "strategy"
            all_chunks.extend(chunks)

    if not all_chunks:
        print("  ⚠️  No chunk files found for BM25 index.")
        _corpus_chunks = []
        _bm25_index = BM25Okapi([[""]])  # empty index
        return _bm25_index, _corpus_chunks

    # Tokenize all chunks for BM25
    tokenized_corpus = [tokenize(c["chunk_text"]) for c in all_chunks]

    _bm25_index = BM25Okapi(tokenized_corpus)
    _corpus_chunks = all_chunks

    return _bm25_index, _corpus_chunks


# ─────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────

def bm25_search(
    query: str,
    top_k: int = 5,
    expansion: str | None = None,
) -> list[dict]:
    """
    Keyword search over rulebook + strategy chunks using BM25.

    Args:
        query:     The user's natural language question.
        top_k:     Number of results to return (default 5).
        expansion: If set, only return chunks from this expansion.

    Returns:
        List of dicts with: expansion, source_page, chunk_text, bm25_score
        Sorted by descending BM25 score.
    """
    index, chunks = _load_index()

    if not chunks:
        return []

    # Tokenize and score
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)

    # Pair scores with chunk data and sort
    scored = list(zip(scores, chunks))

    # Filter by expansion if requested
    if expansion:
        scored = [
            (score, chunk) for score, chunk in scored
            if chunk.get("expansion", "").lower() == expansion.lower()
        ]

    # Sort by score descending and take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[:top_k]

    # Format results (matching the shape of rules_search output)
    results = []
    for score, chunk in top_results:
        if score <= 0:
            continue  # skip zero-score results
        results.append({
            "expansion": chunk.get("expansion", "Unknown"),
            "source_page": chunk.get("source_page", "?"),
            "chunk_text": chunk.get("chunk_text", ""),
            "bm25_score": float(score),
        })

    return results


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "Duration cards",
        "When can I play Reaction cards?",
        "What happens when the supply runs out?",
        "Throne Room combos",
        "trash Copper Estate",
    ]

    # Force index load and print stats
    index, chunks = _load_index()
    print(f"── BM25 Index Stats ──────────────────────")
    print(f"  Total chunks indexed: {len(chunks)}")

    source_types = {}
    for c in chunks:
        st = c.get("source_type", "unknown")
        source_types[st] = source_types.get(st, 0) + 1
    for st, count in sorted(source_types.items()):
        print(f"    {st}: {count}")

    print(f"\n── BM25 Search Tests ─────────────────────")
    for query in test_queries:
        print(f"\n  Q: \"{query}\"")
        results = bm25_search(query, top_k=3)
        if results:
            for r in results:
                score = f"{r['bm25_score']:.3f}"
                exp = r.get("expansion", "?")
                page = r.get("source_page", "?")
                text = r.get("chunk_text", "")[:100]
                print(f"    [{score}] {exp} p.{page}: {text}...")
        else:
            print("    No results.")
