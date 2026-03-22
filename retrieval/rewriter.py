"""
rewriter.py — DominionSage Conversation-Aware Query Rewriter

Resolves pronouns, references, and implicit context in follow-up
questions by examining recent conversation history.

Example:
  History:  Q: "What does Throne Room do?"
            A: "Throne Room lets you play an Action card twice..."
  New query: "What combos well with it?"
  Rewritten: "What combos well with Throne Room?"

Architecture decision: This runs BEFORE the router, so the rest of the
pipeline (router → retrieval → synthesis) never needs to know about
conversation history. It's a preprocessing step — like a SQL view that
normalizes data before the query optimizer sees it.

Cost: One GPT-4o-mini call per query (~$0.0001). If the query is already
self-contained, the model returns it unchanged, so there's no degradation
on first-turn questions.
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


# ─────────────────────────────────────────────────────────────────
# Client (shared singleton pattern, same as synthesizer.py)
# ─────────────────────────────────────────────────────────────────

_openai: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai
    if _openai is None:
        if load_dotenv:
            load_dotenv()
        _openai = OpenAI()
    return _openai


# ─────────────────────────────────────────────────────────────────
# Rewriter prompt
# ─────────────────────────────────────────────────────────────────

REWRITER_SYSTEM = """You are a query rewriter for a Dominion card game assistant.

Your job: take a user's follow-up question and rewrite it as a standalone
question that makes sense WITHOUT any conversation history.

Rules:
1. Replace pronouns (it, them, that, those, this) with the specific
   card names or concepts they refer to from the conversation history.
2. Carry forward any implicit context. If the user asked about a specific
   card and now asks "what about the cost?", rewrite to include the card name.
3. If the question is ALREADY self-contained (no pronouns or references
   to resolve), return it EXACTLY as-is. Do not rephrase or "improve" it.
4. Return ONLY the rewritten question. No explanation, no quotes, no prefix.
5. Keep the rewritten question concise — don't add unnecessary words.

Examples:
  History: Q: "What does Chapel do?" A: "Chapel lets you trash up to 4 cards..."
  New: "Is it good in Big Money?"
  Output: Is Chapel good in Big Money?

  History: Q: "Show me all Duration cards" A: "Here are the Duration cards..."
  New: "Which ones cost 5?"
  Output: Which Duration cards cost 5?

  History: Q: "What does Throne Room do?" A: "Throne Room plays an Action twice."
  New: "What combos well with it?"
  Output: What combos well with Throne Room?

  History: (empty)
  New: "What does Village do?"
  Output: What does Village do?

  History: Q: "Compare Village and Festival" A: "Village gives +2 Actions..."
  New: "When should I buy Festival instead?"
  Output: When should I buy Festival instead of Village?
"""


# ─────────────────────────────────────────────────────────────────
# Core rewriter
# ─────────────────────────────────────────────────────────────────

# How many recent Q&A pairs to include as context.
# 3 is enough to resolve most references without burning too many tokens.
MAX_HISTORY_TURNS = 3


def rewrite_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Rewrite a query to be self-contained using conversation history.

    Args:
        query: The user's new question (may contain pronouns/references).
        conversation_history: List of recent exchanges, each dict with:
            - "role": "user" or "assistant"
            - "content": the message text

    Returns:
        A self-contained version of the query. If no rewriting is needed,
        returns the original query unchanged.
    """
    # No history → nothing to resolve, return as-is
    if not conversation_history:
        return query

    # Quick heuristic: if the query contains no pronouns or short references,
    # skip the LLM call entirely. This saves ~100ms and $0.0001 per query
    # that doesn't need rewriting.
    if not _needs_rewriting(query):
        return query

    # Build the conversation context for the rewriter
    # Take only the last N turns to keep the prompt small
    recent = conversation_history[-(MAX_HISTORY_TURNS * 2):]
    history_text = _format_history(recent)

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": REWRITER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_text}\n\n"
                        f"New question: {query}"
                    ),
                },
            ],
            temperature=0.0,  # deterministic — we want consistent rewrites
            max_tokens=150,   # rewrites should be short
        )
        rewritten = response.choices[0].message.content.strip()

        # Sanity check: if the rewriter returned something empty or
        # suspiciously long, fall back to the original
        if not rewritten or len(rewritten) > len(query) * 3:
            return query

        return rewritten

    except Exception:
        # If the rewriter fails for any reason, the original query
        # still works — it just won't have resolved references.
        # Fail silently so the pipeline keeps working.
        return query


# ─────────────────────────────────────────────────────────────────
# Heuristic: does this query need rewriting?
# ─────────────────────────────────────────────────────────────────

# Pronouns and short references that signal a follow-up question
_FOLLOWUP_SIGNALS = {
    # Pronouns
    "it", "its", "they", "them", "their", "those", "that", "this",
    "these", "the card", "the same",
    # Implicit references
    "which ones", "how about", "what about", "and also",
    "instead", "compared to", "versus", "vs",
    # Continuations
    "more about", "tell me more", "go on", "continue",
    "anything else", "what else",
}


def _needs_rewriting(query: str) -> bool:
    """
    Quick check: does the query contain signals that suggest it's
    a follow-up referencing previous context?

    This is a cheap heuristic to avoid calling the LLM on queries
    that are clearly self-contained (e.g., "What does Chapel do?").
    False positives are fine — the LLM will just return the query
    unchanged. False negatives mean we miss a rewrite opportunity,
    but that's rare with this signal set.
    """
    query_lower = query.lower()
    return any(signal in query_lower for signal in _FOLLOWUP_SIGNALS)


# ─────────────────────────────────────────────────────────────────
# History formatting
# ─────────────────────────────────────────────────────────────────

def _format_history(messages: list[dict]) -> str:
    """
    Format conversation history into a compact string for the rewriter.

    We truncate long assistant responses to keep the prompt small —
    the rewriter only needs to know WHAT was discussed, not the
    full answer text.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            parts.append(f"Q: {content}")
        elif role == "assistant":
            # Truncate long answers — the rewriter just needs the gist
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"A: {content}")

    return "\n".join(parts) if parts else "(No previous conversation)"


# ─────────────────────────────────────────────────────────────────
# Debug / testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test cases for the rewriter
    test_cases = [
        {
            "history": [
                {"role": "user", "content": "What does Throne Room do?"},
                {"role": "assistant", "content": "Throne Room costs 4 and lets you play an Action card from your hand twice."},
            ],
            "query": "What combos well with it?",
            "expected_contains": "Throne Room",
        },
        {
            "history": [
                {"role": "user", "content": "Show me all Duration cards"},
                {"role": "assistant", "content": "Here are the Duration cards: Wharf, Lighthouse, Merchant Ship..."},
            ],
            "query": "Which ones cost 5?",
            "expected_contains": "Duration",
        },
        {
            "history": [],
            "query": "What does Village do?",
            "expected_contains": "Village",
        },
        {
            "history": [
                {"role": "user", "content": "What does Chapel do?"},
                {"role": "assistant", "content": "Chapel lets you trash up to 4 cards from your hand."},
            ],
            "query": "Is it good in Big Money decks?",
            "expected_contains": "Chapel",
        },
    ]

    print("=" * 60)
    print("DominionSage — Query Rewriter Tests")
    print("=" * 60)

    for i, tc in enumerate(test_cases):
        original = tc["query"]
        rewritten = rewrite_query(original, tc["history"])
        has_expected = tc["expected_contains"].lower() in rewritten.lower()

        status = "✅" if has_expected else "❌"
        print(f"\n  Test {i + 1}: {status}")
        print(f"    Original:  {original}")
        print(f"    Rewritten: {rewritten}")
        if not has_expected:
            print(f"    ⚠️  Expected '{tc['expected_contains']}' in rewritten query")

    print(f"\n{'=' * 60}")
