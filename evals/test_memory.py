"""
evals/test_memory.py — DominionSage Multi-Turn Memory Evaluation

Tests the conversation-aware query rewriting pipeline across 5 multi-turn
test cases. Each case is a short conversation (2-4 turns) where later
questions rely on context from earlier ones via pronouns, ellipsis, or
implicit references.

What is being tested:
  1. Rewrite accuracy  — Was the follow-up query correctly resolved?
  2. Routing accuracy  — Did the resolved query reach the right handler?
  3. Answer relevance  — Does the final answer address the original intent?

Design principle: Each test case is written so that the LAST turn is the
one that requires memory to answer correctly. The prior turns are the
"memory" the system must draw on.

Usage:
  python evals/test_memory.py                 # run all 5 cases
  python evals/test_memory.py --verbose       # show full answers
  python evals/test_memory.py --case 2        # run a single case (1-indexed)
  python evals/test_memory.py --judge         # add LLM-as-judge scoring
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.orchestrator import answer_question
from retrieval.rewriter import rewrite_query


# ─────────────────────────────────────────────────────────────────
# Multi-turn test cases
# ─────────────────────────────────────────────────────────────────

MULTI_TURN_CASES = [
    # ── Case 1: Pronoun resolution (card → "it") ──────────────────
    {
        "id": 1,
        "name": "Pronoun → card name (Chapel strategy)",
        "description": (
            "User asks about Chapel, then asks a strategy question using 'it'. "
            "The system must resolve 'it' to 'Chapel' before routing."
        ),
        "turns": [
            {
                "role": "user",
                "content": "What does Chapel do?",
                # This turn is simulated — we call answer_question, but its
                # answer becomes the assistant message in history.
            },
        ],
        "final_turn": {
            "question": "Is it worth buying early?",
            "expected_rewrite_contains": "Chapel",
            "expected_query_type":       "strategy_combo",
            "key_facts": ["trash", "early"],
            "notes": (
                "'it' must resolve to Chapel. The rewritten query should be "
                "something like 'Is Chapel worth buying early?' and should "
                "route as strategy_combo."
            ),
        },
    },

    # ── Case 2: "Which ones" reference to a filtered set ──────────
    {
        "id": 2,
        "name": "Set reference — 'which ones' from filtered search",
        "description": (
            "User requests all Duration cards, then asks 'which ones cost 5?' "
            "The system must understand 'which ones' refers to Duration cards."
        ),
        "turns": [
            {
                "role": "user",
                "content": "List all Duration cards from Seaside",
            },
        ],
        "final_turn": {
            "question": "Which ones cost 5?",
            "expected_rewrite_contains": "Duration",
            "expected_query_type":       "filtered_search",
            "key_facts": ["Wharf", "Merchant Ship"],
            "notes": (
                "'which ones' must resolve to 'Duration cards from Seaside'. "
                "Routing should stay as filtered_search with cost=5 filter."
            ),
        },
    },

    # ── Case 3: Multi-hop comparison ("that card" after comparison) ─
    {
        "id": 3,
        "name": "Multi-hop reference — follow-up after comparison",
        "description": (
            "User compares Smithy vs Laboratory for draw, then asks about 'the cheaper one'. "
            "The system must identify Smithy (cost 4) as cheaper than Laboratory (cost 5)."
        ),
        "turns": [
            {
                "role": "user",
                "content": "Is Smithy or Laboratory better for drawing cards?",
            },
        ],
        "final_turn": {
            "question": "What combos well with the cheaper one?",
            "expected_rewrite_contains": "Smithy",
            "expected_query_type":       "strategy_combo",
            "key_facts": ["Smithy", "action"],
            "notes": (
                "'the cheaper one' should resolve to Smithy (costs 4 vs Lab's 5). "
                "Answer should be a strategy/combo answer about Smithy."
            ),
        },
    },

    # ── Case 4: Two-turn build-up (expansion + type narrowing) ─────
    {
        "id": 4,
        "name": "Two-turn context build-up — Nocturne Night cards",
        "description": (
            "User establishes the expansion context (Nocturne), then asks about a "
            "specific card type using 'those', requiring both pieces of context."
        ),
        "turns": [
            {
                "role": "user",
                "content": "Tell me about Nocturne cards",
            },
        ],
        "final_turn": {
            "question": "How do those Night cards actually work mechanically?",
            "expected_rewrite_contains": "Nocturne",
            "expected_query_type":       "rules_question",
            "key_facts": ["night", "action"],
            "notes": (
                "'those Night cards' references Nocturne from context. "
                "Should route as a rules_question about Night card mechanics."
            ),
        },
    },

    # ── Case 5: Three-turn conversation with accumulated context ────
    {
        "id": 5,
        "name": "Three-turn conversation — Throne Room + Village combo",
        "description": (
            "User learns about Throne Room and Village across two setup turns, "
            "then asks 'What combos well with both of them?' — 'them' must resolve "
            "to Throne Room and Village from the conversation history."
        ),
        "turns": [
            {
                "role": "user",
                "content": "What does Throne Room do?",
            },
            {
                "role": "user",
                "content": "And Village?",
            },
        ],
        "final_turn": {
            "question": "What combos well with both of them?",
            "expected_rewrite_contains": "Throne Room",
            "expected_query_type":       "strategy_combo",
            "key_facts": ["Throne Room", "Village", "action"],
            "notes": (
                "'them' must resolve to Throne Room + Village from earlier in the "
                "conversation. The rewrite should mention at least Throne Room. "
                "Route as strategy_combo since it's asking about combos."
                "\n\nNOTE: Ordinal-reference phrasing ('the first one', 'the second')"
                " does not trigger the rewriter because those words are not in"
                " _FOLLOWUP_SIGNALS in rewriter.py — a known limitation."
            ),
        },
    },
]


# ─────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────

RATE_LIMIT_DELAY = 1.5  # seconds between API calls


def simulate_turn(question: str, history: list[dict], verbose: bool = False) -> dict:
    """
    Call answer_question with the given history and return the full result.
    Appends both the user question and assistant response to history in-place.
    """
    result = answer_question(question, conversation_history=history)

    # Add to history for subsequent turns
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})

    if verbose:
        print(f"      Q: {question}")
        print(f"      A: {result['answer'][:120]}...")
        if "rewritten_query" in result:
            print(f"      ↳ Rewritten: {result['rewritten_query']}")

    return result


# ─────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────

def score_rewrite(rewritten_query: str, expected_substring: str) -> bool:
    """Did the rewriter include the expected card/concept name?"""
    return expected_substring.lower() in rewritten_query.lower()


def score_routing(actual_type: str, expected_type: str) -> bool:
    """Did the final turn route to the correct query handler?"""
    return actual_type == expected_type


def score_answer_quality(answer: str, key_facts: list[str]) -> float:
    """What fraction of expected key facts appear in the answer?"""
    if not key_facts:
        return 1.0
    found = [f for f in key_facts if f.lower() in answer.lower()]
    return len(found) / len(key_facts)


# ─────────────────────────────────────────────────────────────────
# LLM-as-judge (optional)
# ─────────────────────────────────────────────────────────────────

MEMORY_JUDGE_PROMPT = """You are an expert evaluator for a conversational AI Dominion card game assistant.

You are evaluating whether the assistant correctly used CONVERSATION MEMORY to answer a follow-up question.

The user asked a series of questions. The FINAL question is a follow-up that references something from an earlier message.

Evaluate ONLY the final answer on two criteria:

1. MEMORY RESOLUTION (1-5):
   Did the system correctly identify what the user was referring to?
   - Score 5: Perfectly identified the referenced card/concept from context
   - Score 3: Partially resolved the reference (e.g., right card, wrong detail)
   - Score 1: Failed — answered a different card/concept or ignored the history

2. ANSWER QUALITY (1-5):
   Given the correctly resolved card/concept, is the answer actually helpful?
   - Score 5: Accurate, complete, addresses the user's real intent
   - Score 3: Partially correct or incomplete
   - Score 1: Wrong facts or clearly missing the point

RESPOND IN EXACTLY THIS JSON FORMAT (no other text):
{
  "memory_resolution": <1-5>,
  "answer_quality": <1-5>,
  "resolved_reference_as": "what the model resolved the reference to (or 'failed')",
  "errors": ["list any errors"],
  "reasoning": "brief explanation"
}"""


def llm_memory_judge(
    conversation_history: list[dict],
    final_question: str,
    final_answer: str,
    expected_rewrite_contains: str,
) -> dict:
    """Use GPT-4o to evaluate whether the memory resolution worked correctly."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        return {"memory_resolution": 0, "answer_quality": 0, "errors": [str(e)], "reasoning": "OpenAI unavailable"}

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
        for m in conversation_history
    )

    user_message = f"""CONVERSATION HISTORY (before the final question):
{history_text}

FINAL QUESTION (the follow-up requiring memory): {final_question}

ASSISTANT'S FINAL ANSWER:
{final_answer}

EXPECTED REFERENCE TARGET: The follow-up refers to "{expected_rewrite_contains}".

Grade the assistant's memory resolution and answer quality."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": MEMORY_JUDGE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(result_text)
        return scores
    except Exception as e:
        return {
            "memory_resolution": 0,
            "answer_quality": 0,
            "resolved_reference_as": "unknown",
            "errors": [str(e)],
            "reasoning": "Judge call failed",
        }


# ─────────────────────────────────────────────────────────────────
# Single case runner
# ─────────────────────────────────────────────────────────────────

def run_case(case: dict, verbose: bool = False, use_judge: bool = False) -> dict:
    """
    Simulate a full multi-turn conversation for a single test case.

    1. Simulate all setup turns to build conversation history.
    2. Run the final turn with that history.
    3. Score rewrite accuracy, routing accuracy, and answer quality.
    """
    case_id   = case["id"]
    case_name = case["name"]
    final     = case["final_turn"]

    if verbose:
        print(f"\n  ── Setup turns for Case {case_id}: \"{case_name}\" ──")

    # Build conversation history by simulating each setup turn
    history: list[dict] = []
    try:
        for turn in case["turns"]:
            simulate_turn(turn["content"], history, verbose=verbose)
            time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        return {
            "case_id":            case_id,
            "case_name":          case_name,
            "error":              f"Setup turn failed: {e}",
            "rewrite_correct":    False,
            "routing_correct":    False,
            "answer_quality":     0.0,
            "rewritten_query":    "",
            "actual_query_type":  "error",
        }

    # Run the final (memory-dependent) turn
    final_question = final["question"]
    if verbose:
        print(f"\n  ── Final turn (memory-dependent) ──")

    try:
        result = answer_question(final_question, conversation_history=history)
    except Exception as e:
        return {
            "case_id":           case_id,
            "case_name":         case_name,
            "error":             f"Final turn failed: {e}",
            "rewrite_correct":   False,
            "routing_correct":   False,
            "answer_quality":    0.0,
            "rewritten_query":   "",
            "actual_query_type": "error",
        }

    rewritten_query  = result.get("rewritten_query", final_question)
    actual_type      = result["query_type"]
    answer           = result["answer"]

    # Scores
    rewrite_ok  = score_rewrite(rewritten_query, final["expected_rewrite_contains"])
    routing_ok  = score_routing(actual_type, final["expected_query_type"])
    quality     = score_answer_quality(answer, final.get("key_facts", []))

    key_facts   = final.get("key_facts", [])
    facts_found = [f for f in key_facts if f.lower() in answer.lower()]
    facts_missed = [f for f in key_facts if f.lower() not in answer.lower()]

    row = {
        "case_id":            case_id,
        "case_name":          case_name,
        "setup_turns":        len(case["turns"]),
        "final_question":     final_question,
        "rewritten_query":    rewritten_query,
        "expected_rewrite":   final["expected_rewrite_contains"],
        "rewrite_correct":    rewrite_ok,
        "expected_type":      final["expected_query_type"],
        "actual_query_type":  actual_type,
        "routing_correct":    routing_ok,
        "answer_quality":     round(quality, 2),
        "facts_found":        "; ".join(facts_found),
        "facts_missed":       "; ".join(facts_missed),
        "error":              "",
    }

    if verbose:
        print(f"      Final Q: {final_question}")
        print(f"      Rewritten to: {rewritten_query}")
        print(f"      Type: {actual_type}  (expected: {final['expected_query_type']})")
        print(f"      Answer: {answer[:160]}...")
        if facts_missed:
            print(f"      Missed key facts: {facts_missed}")

    # LLM judge (optional)
    if use_judge:
        judge = llm_memory_judge(
            conversation_history=history,
            final_question=final_question,
            final_answer=answer,
            expected_rewrite_contains=final["expected_rewrite_contains"],
        )
        row["judge_memory_resolution"] = judge.get("memory_resolution", 0)
        row["judge_answer_quality"]    = judge.get("answer_quality", 0)
        row["judge_resolved_as"]       = judge.get("resolved_reference_as", "")
        row["judge_errors"]            = "; ".join(judge.get("errors", []))
        row["judge_reasoning"]         = judge.get("reasoning", "")

        if verbose:
            print(f"      Judge — Memory: {judge.get('memory_resolution')}/5  "
                  f"Quality: {judge.get('answer_quality')}/5")
            print(f"      Judge resolved reference as: \"{judge.get('resolved_reference_as')}\"")

    return row


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    total = len(results)
    if total == 0:
        print("No results.")
        return

    rewrite_acc = sum(r["rewrite_correct"] for r in results) / total
    routing_acc = sum(r["routing_correct"] for r in results) / total
    avg_quality = sum(r["answer_quality"] for r in results) / total
    has_judge   = "judge_memory_resolution" in results[0]

    print(f"\n{'=' * 62}")
    print(f"  MEMORY EVAL RESULTS — {total} multi-turn test cases")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 62}")
    print(f"  Rewrite accuracy:   {rewrite_acc:>6.0%}  {'✅' if rewrite_acc >= 0.80 else '⚠️'} (target: ≥80%)")
    print(f"  Routing accuracy:   {routing_acc:>6.0%}  {'✅' if routing_acc >= 0.80 else '⚠️'} (target: ≥80%)")
    print(f"  Answer quality:     {avg_quality:>6.0%}  {'✅' if avg_quality >= 0.75 else '⚠️'} (target: ≥75%)")

    if has_judge:
        avg_mem_res  = sum(r.get("judge_memory_resolution", 0) for r in results) / total
        avg_ans_qual = sum(r.get("judge_answer_quality", 0) for r in results) / total
        print(f"\n  ── LLM Judge Scores (GPT-4o, 1-5 scale) ──")
        print(f"  Memory resolution:  {avg_mem_res:>5.1f}  {'✅' if avg_mem_res >= 4 else '⚠️' if avg_mem_res >= 3 else '❌'}")
        print(f"  Answer quality:     {avg_ans_qual:>5.1f}  {'✅' if avg_ans_qual >= 4 else '⚠️' if avg_ans_qual >= 3 else '❌'}")

    # Per-case details
    print(f"\n  {'ID':<4} {'Rewrite':<9} {'Route':<8} {'Quality':<9} {'Case Name'}")
    print(f"  {'─' * 58}")
    for r in results:
        rw = "✅" if r["rewrite_correct"] else "❌"
        rt = "✅" if r["routing_correct"] else "❌"
        q  = r["answer_quality"]
        qi = "✅" if q >= 0.75 else "⚠️" if q >= 0.5 else "❌"
        j_col = ""
        if has_judge:
            jm = r.get("judge_memory_resolution", 0)
            ji = "✅" if jm >= 4 else "⚠️" if jm >= 3 else "❌"
            j_col = f"  Judge:{ji}({jm}/5)"
        print(f"  #{r['case_id']:<3} {rw:<9} {rt:<8} {qi}({q:.0%})    {r['case_name'][:35]}{j_col}")
        if r.get("error"):
            print(f"       ⚠️  Error: {r['error']}")
        if r.get("facts_missed"):
            print(f"       Missed: {r['facts_missed']}")
        if has_judge and r.get("judge_resolved_as"):
            print(f"       Resolved reference as: \"{r['judge_resolved_as']}\"")

    # Failures
    failures = [r for r in results if not r["rewrite_correct"] or not r["routing_correct"] or r["answer_quality"] < 0.5]
    if failures:
        print(f"\n  ── Cases to Investigate ──")
        for r in failures:
            issues = []
            if not r["rewrite_correct"]:
                issues.append(f"rewrite missing '{r['expected_rewrite']}' → got: \"{r['rewritten_query']}\"")
            if not r["routing_correct"]:
                issues.append(f"routed as {r['actual_query_type']} (expected {r['expected_type']})")
            if r["answer_quality"] < 0.5:
                issues.append(f"low quality ({r['answer_quality']:.0%}), missed: {r['facts_missed']}")
            print(f"  ⚠️  Case #{r['case_id']}: \"{r['case_name']}\"")
            for issue in issues:
                print(f"      → {issue}")

    # Improvement suggestions
    print(f"\n  ── Improvement Suggestions ──")
    suggestions = []
    if rewrite_acc < 0.80:
        suggestions.append("Rewrite: Expand _FOLLOWUP_SIGNALS in rewriter.py or add more example patterns to REWRITER_SYSTEM.")
    if routing_acc < 0.80:
        suggestions.append("Routing: The rewritten query may be malformed — check the rewritten_query strings above.")
    if avg_quality < 0.75:
        suggestions.append("Quality: The right context may not be reaching the synthesizer — check retrieval after rewriting.")
    if not suggestions:
        print("  🎉 All memory eval metrics meet targets!")
    else:
        for s in suggestions:
            print(f"  📌 {s}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run DominionSage multi-turn memory evaluation."
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full conversation turns and answers.")
    parser.add_argument("--judge", "-j", action="store_true",
                        help="Enable LLM-as-judge scoring (GPT-4o, ~$0.10 per case).")
    parser.add_argument("--case", "-c", type=int, default=None,
                        help="Run a single test case by ID (1-indexed, 1–5).")
    args = parser.parse_args()

    print("=" * 62)
    print("  DominionSage — Multi-Turn Memory Evaluation")
    if args.judge:
        print("  (LLM-as-Judge enabled — GPT-4o will grade memory resolution)")
    print("=" * 62)

    cases = MULTI_TURN_CASES
    if args.case is not None:
        matching = [c for c in cases if c["id"] == args.case]
        if not matching:
            print(f"Error: No case with ID {args.case} (valid range: 1–{len(cases)})")
            sys.exit(1)
        cases = matching

    results = []

    for case in cases:
        print(f"\n  ▶ Case #{case['id']}: {case['name']}")
        print(f"    {case['description']}")
        print(f"    Setup turns: {len(case['turns'])}  |  Final: \"{case['final_turn']['question']}\"")

        row = run_case(case, verbose=args.verbose, use_judge=args.judge)
        results.append(row)

        # Inline result
        rw_icon = "✅" if row["rewrite_correct"] else "❌"
        rt_icon = "✅" if row["routing_correct"] else "❌"
        q       = row["answer_quality"]
        q_icon  = "✅" if q >= 0.75 else "⚠️" if q >= 0.5 else "❌"

        rewrite_display = row["rewritten_query"] or "(unchanged)"
        print(f"\n    Result:")
        print(f"      Rewrite:  {rw_icon} \"{rewrite_display}\"")
        print(f"      Route:    {rt_icon} {row['actual_query_type']} (expected: {row['expected_type']})")
        print(f"      Quality:  {q_icon} {q:.0%}")
        if row.get("error"):
            print(f"      ⚠️  Error: {row['error']}")

        # Pause between cases to respect API rate limits
        if case != cases[-1]:
            time.sleep(2.0)

    print_summary(results)


if __name__ == "__main__":
    main()
