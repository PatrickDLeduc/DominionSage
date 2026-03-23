"""
DominionSage Retrieval Package

The retrieval layer is the core of the RAG system. It contains:
  - router.py        — classifies queries into 4 types
  - card_lookup.py   — SQL queries against the cards table
  - rules_search.py  — vector similarity search over rulebook chunks
  - bm25_search.py   — BM25 keyword search over local chunk files
  - hybrid_search.py — RRF fusion of vector + BM25 results
  - orchestrator.py  — coordinates retrieval and calls the synthesizer
  - synthesizer.py   — generates final answers via GPT-4o-mini
"""

from retrieval.orchestrator import answer_question

__all__ = ["answer_question"]
