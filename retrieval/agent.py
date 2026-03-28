"""
agent.py — DominionSage LLM Agent with Tool Use

Replaces the rigid rule-based router with an LLM agent that dynamically
decides which tools to call. The agent can call multiple tools per query,
reason about intermediate results, and synthesize a final answer.

Uses OpenAI native function calling (no external framework needed).
"""

import json
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

from retrieval.rewriter import rewrite_query
from retrieval.synthesizer import STRATEGY_PRINCIPLES, format_context
from retrieval.models import SynthesizerResponse
from retrieval.tools import TOOL_SCHEMAS, execute_tool


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
# Agent system prompt
# ─────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = f"""You are DominionSage, an expert assistant for the Dominion card game.

You have access to tools for looking up cards, searching rules, and finding strategy advice.
Use the appropriate tool(s) to gather context before answering the user's question.
You may call multiple tools if the question spans different areas (e.g., card data + strategy).

RULES:
1. ALWAYS use at least one tool before answering — do not answer from memory alone.
2. DECLINE OFF-TOPIC QUESTIONS: If the user asks something unrelated to Dominion, politely decline.
3. STAY GROUNDED: Only use information returned by your tools. If the tools don't have enough info, say so honestly.
4. BE CONCISE: Give thorough but focused answers. Bold card names when first mentioned.
5. After gathering context via tools, provide your final answer as plain text (not JSON).

{STRATEGY_PRINCIPLES}"""


# ─────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────

def _run_agent_loop(
    query: str,
    conversation_history: list[dict] | None = None,
    kingdom_context: str | None = None,
    max_iterations: int = 5,
    on_tool_call: callable = None,
) -> tuple[str, list[dict]]:
    """
    Run the agent loop: call tools as needed, then return the final answer.

    Args:
        query:                The user's (possibly rewritten) question.
        conversation_history: Last few conversation turns for context.
        kingdom_context:      Active kingdom cards string, if any.
        max_iterations:       Max tool-call rounds to prevent runaway loops.
        on_tool_call:         Optional callback(tool_name, args) for UI progress.

    Returns:
        (answer_text, all_sources) where all_sources is accumulated from tool calls.
    """
    client = _get_client()
    all_sources: list[dict] = []

    # Build initial messages
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    # Add recent conversation history (last 3 turns)
    if conversation_history:
        for msg in conversation_history[-6:]:  # 3 turns = 6 messages
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Build user message with optional kingdom context
    user_content = query
    if kingdom_context:
        user_content += (
            f"\n\n[Active Kingdom Cards: {kingdom_context}. "
            f"Consider these cards in your answer if relevant.]"
        )
    messages.append({"role": "user", "content": user_content})

    # Agent loop
    for iteration in range(max_iterations):
        # First iteration: force at least one tool call
        tool_choice = "required" if iteration == 0 else "auto"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice,
            temperature=0.3,
            max_tokens=1500,
        )

        choice = response.choices[0]

        # If the model wants to call tools, execute them
        if choice.message.tool_calls:
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                if on_tool_call:
                    on_tool_call(tool_name, arguments)

                text_result, sources = execute_tool(tool_name, arguments)
                all_sources.extend(sources)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": text_result,
                })
        else:
            # No tool calls — agent is done, return the text answer
            return choice.message.content or "", all_sources

    # If we hit max iterations, the last response should have content
    last_content = response.choices[0].message.content
    if last_content:
        return last_content, all_sources

    # Fallback: ask the model to synthesize from what it has
    messages.append({"role": "user", "content": "Please provide your final answer based on the information gathered."})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
    )
    return response.choices[0].message.content or "", all_sources


# ─────────────────────────────────────────────────────────────────
# Structured citation pass
# ─────────────────────────────────────────────────────────────────

def _extract_citations(answer_text: str, sources: list[dict]) -> dict:
    """
    Take the agent's raw answer and accumulated sources, produce structured
    citations using the same SynthesizerResponse schema as the pipeline.

    Returns dict with answer_text, citations, and source_map.
    """
    context_text, source_map = format_context(sources)

    if not source_map:
        return {
            "answer_text": answer_text,
            "citations": [],
            "source_map": {},
        }

    client = _get_client()

    source_labels = ", ".join(source_map.keys())
    prompt = (
        f"Below is an answer about Dominion and the sources that were used. "
        f"Rewrite the answer lightly if needed for clarity, and produce structured citations.\n\n"
        f"Available source labels: {source_labels}\n\n"
        f"Sources:\n{context_text}\n\n"
        f"Answer to cite:\n{answer_text}"
    )

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a citation assistant. Given an answer and its sources, "
                    "produce the answer with structured citations. Do NOT embed "
                    "[Source N] labels in the answer text. Use the citations field "
                    "to map claims to source labels."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
            response_format=SynthesizerResponse,
        )

        parsed = response.choices[0].message.parsed
        return {
            "answer_text": parsed.answer,
            "citations": [c.model_dump() for c in parsed.citations],
            "source_map": source_map,
        }
    except Exception:
        # Fallback: return the answer without structured citations
        return {
            "answer_text": answer_text,
            "citations": [],
            "source_map": source_map,
        }


# ─────────────────────────────────────────────────────────────────
# Public entry point (matches orchestrator.answer_question signature)
# ─────────────────────────────────────────────────────────────────

def answer_question_agent(
    query: str,
    expansion: str | None = None,
    conversation_history: list[dict] | None = None,
    kingdom_context: str | None = None,
    limit_cards: bool = True,
    on_tool_call: callable = None,
) -> dict:
    """
    Agent-based question answering. Drop-in replacement for
    orchestrator.answer_question() with the same return dict shape.

    Args:
        query:                The user's natural language question.
        expansion:            Optional expansion filter (passed to tools).
        conversation_history: Recent messages for pronoun resolution.
        kingdom_context:      Active kingdom cards string.
        limit_cards:          (Kept for signature compatibility, not used by agent.)
        on_tool_call:         Optional callback(tool_name, args) for UI progress.

    Returns:
        Dict matching orchestrator.answer_question() shape:
        {answer, citations, source_map, meta_notes, sources, query_type,
         original_query?, rewritten_query?}
    """
    # Step 1: Rewrite query to resolve pronouns
    original_query = query
    query = rewrite_query(query, conversation_history)

    # Step 2: Run the agent loop
    answer_text, all_sources = _run_agent_loop(
        query=query,
        conversation_history=conversation_history,
        kingdom_context=kingdom_context,
        on_tool_call=on_tool_call,
    )

    # Step 3: Structured citation pass
    synth_result = _extract_citations(answer_text, all_sources)

    # Step 4: Extract meta notes
    meta_notes = [
        s["data"]["note"]
        for s in all_sources
        if s["type"] == "meta"
    ]

    result = {
        "answer": synth_result["answer_text"],
        "citations": synth_result.get("citations", []),
        "source_map": synth_result.get("source_map", {}),
        "meta_notes": meta_notes,
        "sources": all_sources,
        "query_type": "agent",
    }

    if query != original_query:
        result["original_query"] = original_query
        result["rewritten_query"] = query

    return result
