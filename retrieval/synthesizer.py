"""
synthesizer.py — DominionSage Answer Generator (Phase 3)

Takes retrieved context (card data + rulebook chunks) and the user's
original question, then calls GPT-4o-mini to generate a final answer
with source citations.

Architecture decision: GPT-4o-mini over GPT-4o.
  The synthesis step doesn't require frontier-level reasoning. What
  matters most is the RETRIEVAL quality — whether the right context
  was found. A cheaper, faster model is fine for assembling that
  context into a coherent answer.

  Analogy: The retrieval layer is the research assistant who finds
  the right books and highlights the relevant passages. The synthesizer
  is the writer who turns those highlights into a clear paragraph.
  You need a great research assistant, but the writer just needs to
  be competent.
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
# Client
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
# System prompt
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DominionSage, an expert assistant for the Dominion card game.

Your job is to answer the user's question using ONLY the provided context.
Follow these rules:

1. CITE YOUR SOURCES: Always mention card names and rulebook page numbers
   when referencing specific information. Example: "According to the Base
   rulebook (p.7), Reaction cards can be revealed when..."

2. STAY GROUNDED: Only use information from the provided context. If the
   context doesn't contain enough information to fully answer the question,
   say so honestly and explain what you DO know from the context.

3. BE CONCISE: Give thorough but focused answers. Don't pad with
   unnecessary preamble or caveats.

4. FORMAT CLEARLY: Use card names in bold when first mentioned. If listing
   multiple cards, organize them logically (by cost, by function, etc.).

5. DISTINGUISH SOURCES: If your answer draws from both card data and
   rulebook text, make it clear which information comes from where.

6. IMPORTANT NOTES: If the context includes an [IMPORTANT NOTE], you MUST
   mention it in your answer. These notes contain critical information like
   result limits that the user needs to know about."""


# ─────────────────────────────────────────────────────────────────
# Context formatting
# ─────────────────────────────────────────────────────────────────

def format_context(sources: list[dict]) -> str:
    """
    Format retrieved sources into a context string for the LLM.

    Each source is tagged with its type (card DB or rulebook) so the
    LLM can cite them properly. This is the "data lineage" of the
    AI answer — the LLM can see exactly where each fact came from.
    """
    parts = []

    for source in sources:
        if source["type"] == "card_db":
            card = source["data"]
            parts.append(
                f"[SOURCE: Card Database — {card['name']}]\n"
                f"  Name: {card['name']}\n"
                f"  Cost: {card.get('cost', '?')}\n"
                f"  Type: {card.get('type', '?')}\n"
                f"  Expansion: {card.get('expansion', '?')}\n"
                f"  Text: {card.get('text', 'N/A')}\n"
                f"  +Actions: {card.get('plus_actions', 0)} | "
                f"+Cards: {card.get('plus_cards', 0)} | "
                f"+Buys: {card.get('plus_buys', 0)} | "
                f"+Coins: {card.get('plus_coins', 0)}"
            )

        elif source["type"] == "rulebook":
            chunk = source["data"]
            similarity = chunk.get("similarity", "")
            sim_str = f" (relevance: {similarity:.2f})" if similarity else ""
            parts.append(
                f"[SOURCE: Rulebook — {chunk.get('expansion', '?')} "
                f"p.{chunk.get('source_page', '?')}{sim_str}]\n"
                f"{chunk.get('chunk_text', 'N/A')}"
            )

        elif source["type"] == "meta":
            parts.append(
                f"[IMPORTANT NOTE: {source['data'].get('note', '')}]"
            )

    if not parts:
        return "(No context available — answer based on general knowledge.)"

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# Synthesis
# ─────────────────────────────────────────────────────────────────

def synthesize_answer(query: str, context: dict) -> str:
    """
    Generate a final answer using retrieved context.

    Args:
        query:   The user's original question.
        context: Dict with "query_type" and "sources" list.

    Returns:
        The LLM's generated answer string.
    """
    client = _get_client()
    context_text = format_context(context.get("sources", []))
    query_type = context.get("query_type", "unknown")

    # Add a type-specific hint to help the LLM format its answer
    type_hints = {
        "card_lookup": "The user is asking about a specific card. Lead with the card's key information.",
        "filtered_search": "The user wants a list of cards matching criteria. Present results in a clear, organized way.",
        "rules_question": "The user has a rules question. Be precise and cite the rulebook page.",
        "strategy_combo": "The user wants strategy advice. Explain the reasoning behind your suggestions.",
    }
    hint = type_hints.get(query_type, "")

    user_message = f"""Context:
{context_text}

---

Question: {query}

{hint}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,  # Low temperature for factual accuracy
            max_tokens=1000,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Sorry, I encountered an error generating the answer: {str(e)}"
