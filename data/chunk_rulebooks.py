"""
chunk_rulebooks.py — DominionSage Rulebook Chunker (Phase 2)

Extracts text from all rulebook PDFs in data/rulebooks/, splits them into
overlapping token-based chunks, and saves to data/rulebook_chunks.json.

The chunking strategy:
  - ~400 tokens per chunk (balances relevance vs. context)
  - 50-token overlap (ensures concepts split across boundaries appear in ≥1 chunk)
  - Expansion name inferred from the PDF filename

Think of chunks like index entries in a search engine. Too large and you
get irrelevant padding around the useful sentence. Too small and you lose
the surrounding context that makes the sentence make sense. 400 tokens
is a standard starting point — your evals in Phase 5 will tell you if
you need to tune it.

Prerequisites:
  pip install pdfplumber tiktoken

Usage:
  python data/chunk_rulebooks.py                        # chunk all PDFs
  python data/chunk_rulebooks.py --chunk-size 300       # smaller chunks
  python data/chunk_rulebooks.py --overlap 100          # more overlap
  python data/chunk_rulebooks.py --dir path/to/pdfs     # custom PDF directory
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Missing dependency: pip install pdfplumber")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("Missing dependency: pip install tiktoken")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 400    # tokens per chunk
DEFAULT_OVERLAP = 50        # overlap tokens between chunks
DEFAULT_PDF_DIR = Path("data/rulebooks")
OUTPUT_PATH = Path("data/rulebook_chunks.json")

# Tokenizer for OpenAI's embedding model
enc = tiktoken.encoding_for_model("text-embedding-3-small")


# ─────────────────────────────────────────────────────────────────
# Expansion name inference
# ─────────────────────────────────────────────────────────────────

# Maps common filename fragments to canonical expansion names.
# Add more entries here as you add rulebooks.
EXPANSION_MAP = {
    "base":         "Base",
    "dominion":     "Base",
    "intrigue":     "Intrigue",
    "seaside":      "Seaside",
    "alchemy":      "Alchemy",
    "prosperity":   "Prosperity",
    "cornucopia":   "Cornucopia",
    "hinterlands":  "Hinterlands",
    "dark ages":    "Dark Ages",
    "darkages":     "Dark Ages",
    "dark_ages":    "Dark Ages",
    "guilds":       "Guilds",
    "adventures":   "Adventures",
    "empires":      "Empires",
    "nocturne":     "Nocturne",
    "renaissance":  "Renaissance",
    "menagerie":    "Menagerie",
    "allies":       "Allies",
    "plunder":      "Plunder",
    "rising sun":   "Rising Sun",
    "risingsun":    "Rising Sun",
    "rising_sun":   "Rising Sun",
}


def infer_expansion(filename: str) -> str:
    """
    Guess the expansion name from the PDF filename.
    Falls back to the filename stem (without extension) if no match found.
    """
    name_lower = filename.lower().replace("-", " ").replace("_", " ")

    for fragment, expansion in EXPANSION_MAP.items():
        if fragment in name_lower:
            return expansion

    # Fallback: clean up the filename and use it as the expansion name
    stem = Path(filename).stem
    # Remove common suffixes like "rulebook", "rules", "2e", etc.
    cleaned = re.sub(r"([-_ ]?(rule ?book|rules|2e|2nd|edition|instructions))+", "", stem, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" -_")
    return cleaned.title() if cleaned else stem.title()


# ─────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from each page of a PDF using pdfplumber.

    pdfplumber is used over PyPDF2 because it handles multi-column
    layouts and complex formatting much better — important for game
    rulebooks which often have sidebars, examples, and card references.

    Returns a list of {page: int, text: str} dicts.
    """
    pages = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Clean up common PDF extraction artifacts
                    text = clean_extracted_text(text)
                    pages.append({
                        "page": i + 1,
                        "text": text,
                    })
    except Exception as e:
        print(f"  ⚠️  Error reading {pdf_path.name}: {e}")
        return []

    return pages


def clean_extracted_text(text: str) -> str:
    """
    Clean up common artifacts from PDF text extraction.
    Rulebooks tend to have headers/footers, page numbers, and
    weird whitespace from multi-column layouts.
    """
    # Collapse multiple spaces (common in multi-column extraction)
    text = re.sub(r" {3,}", "  ", text)

    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove isolated page numbers on their own line
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


# ─────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Split page text into overlapping chunks of ~chunk_size tokens.

    The overlap ensures that a concept split across a chunk boundary
    still appears in at least one complete chunk. Think of it like
    a sliding window — each window sees most of what the previous
    one saw, plus a little more.

    Analogy: If chunking is like cutting a book into index cards,
    the overlap is like writing the last sentence of each card at
    the top of the next card, so you never lose context at the cut.
    """
    chunks = []

    for page_data in pages:
        tokens = enc.encode(page_data["text"])

        # Skip very short pages (headers, copyright pages, etc.)
        if len(tokens) < 20:
            continue

        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)

            # Only keep chunks with meaningful content
            if len(chunk_text.strip()) > 30:
                chunks.append({
                    "source_page": page_data["page"],
                    "chunk_text": chunk_text.strip(),
                    "token_count": len(chunk_tokens),
                })

            start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """Find all PDF files in the given directory."""
    if not pdf_dir.exists():
        print(f"Error: Directory '{pdf_dir}' not found.")
        print(f"Create it and add your rulebook PDFs there:")
        print(f"  mkdir -p {pdf_dir}")
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Error: No PDF files found in '{pdf_dir}'.")
        print(f"Download the Dominion rulebooks and save them there.")
        sys.exit(1)

    return pdfs


def main():
    parser = argparse.ArgumentParser(
        description="Extract and chunk Dominion rulebook PDFs."
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Tokens per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help=f"Overlap tokens between chunks (default: {DEFAULT_OVERLAP})")
    parser.add_argument("--dir", type=Path, default=DEFAULT_PDF_DIR,
                        help=f"Directory containing rulebook PDFs (default: {DEFAULT_PDF_DIR})")
    args = parser.parse_args()

    print("=" * 50)
    print("DominionSage — Rulebook Chunker")
    print("=" * 50)
    print(f"  Chunk size: {args.chunk_size} tokens")
    print(f"  Overlap:    {args.overlap} tokens")
    print(f"  PDF dir:    {args.dir}")

    # Discover PDFs
    pdfs = discover_pdfs(args.dir)
    print(f"\n  Found {len(pdfs)} PDF(s):")
    for pdf in pdfs:
        print(f"    {pdf.name}")

    # Process each PDF
    all_chunks = []
    stats = []

    for pdf_path in pdfs:
        expansion = infer_expansion(pdf_path.name)
        print(f"\n── Processing: {pdf_path.name} → '{expansion}' ──")

        # Extract text
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            print(f"  ⚠️  No text extracted from {pdf_path.name}. Skipping.")
            continue
        print(f"  Extracted text from {len(pages)} pages.")

        # Chunk
        chunks = chunk_pages(pages, args.chunk_size, args.overlap)
        print(f"  Created {len(chunks)} chunks.")

        # Tag with expansion
        for chunk in chunks:
            chunk["expansion"] = expansion

        all_chunks.extend(chunks)
        stats.append({
            "file": pdf_path.name,
            "expansion": expansion,
            "pages": len(pages),
            "chunks": len(chunks),
        })

    # Summary
    print(f"\n── Summary ───────────────────────────────")
    total_tokens = sum(c["token_count"] for c in all_chunks)
    print(f"  {'Expansion':<20} {'Pages':>6} {'Chunks':>7}")
    print(f"  {'─' * 35}")
    for s in stats:
        print(f"  {s['expansion']:<20} {s['pages']:>6} {s['chunks']:>7}")
    print(f"  {'─' * 35}")
    print(f"  {'TOTAL':<20} {sum(s['pages'] for s in stats):>6} {len(all_chunks):>7}")
    print(f"\n  Total tokens across all chunks: {total_tokens:,}")

    # Estimate embedding cost
    # text-embedding-3-small: $0.02 per 1M tokens
    est_cost = (total_tokens / 1_000_000) * 0.02
    print(f"  Estimated embedding cost: ${est_cost:.4f}")

    # Save chunks (without token_count — not needed in DB)
    output_chunks = [
        {
            "expansion": c["expansion"],
            "source_page": c["source_page"],
            "chunk_text": c["chunk_text"],
        }
        for c in all_chunks
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved {len(output_chunks)} chunks to {OUTPUT_PATH}")
    print("   Next step: python data/embed_and_load.py")


if __name__ == "__main__":
    main()
