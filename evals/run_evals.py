"""
run_evals.py — DominionSage Evaluation Suite (Phase 5)

Runs all 20 eval questions through the full pipeline and scores them
on three dimensions:

  1. Routing accuracy  — Did the router classify the query correctly?
  2. Retrieval precision — Did the correct source appear in the results?
  3. Answer quality — Does the generated answer contain the key facts?

This is the AI engineering equivalent of unit tests. The difference is
that AI systems are non-deterministic — the same input can produce
different outputs. Evals handle this by testing PROPERTIES ("does the
answer mention trashing?") rather than exact equality.

Usage:
  python evals/run_evals.py                  # run all evals
  python evals/run_evals.py --verbose        # show full answers
  python evals/run_evals.py --type card_lookup  # run only one type
  python evals/run_evals.py --question 3     # run a single question
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.orchestrator import answer_question


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

QUESTIONS_PATH = Path("evals/questions.json")
RESULTS_PATH = Path("evals/results.csv")
RATE_LIMIT_DELAY = 1.5  # seconds between questions (avoid API rate limits)


# ─────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────

def score_routing(expected_type: str, actual_type: str) -> bool:
    """Did the router classify the query into the correct type?"""
    return expected_type == actual_type


def score_retrieval(sources: list[dict], expected_keywords: list[str]) -> bool:
    """
    Did the expected source appear in the retrieved results?

    Checks whether ANY of the expected keywords appear in the
    serialized source data. This is a recall check — "did we find
    the thing we needed?"
    """
    if not expected_keywords:
        return True  # No expected source specified = auto-pass

    # Serialize all source data into one big string for searching
    source_text = ""
    for s in sources:
        if s["type"] == "card_db":
            source_text += " " + json.dumps(s["data"]).lower()
        elif s["type"] == "rulebook":
            source_text += " " + s["data"].get("chunk_text", "").lower()

    # Check if any expected keyword appears
    return any(kw.lower() in source_text for kw in expected_keywords)


def normalize_text(text: str) -> str:
    """
    Normalize text for fuzzy key_fact matching.
    Handles hyphens, extra spaces, and case differences.
    'Clean-up' and 'cleanup' and 'clean up' all become 'clean up'.
    """
    import re
    text = text.lower()
    text = text.replace("-", " ")       # clean-up → clean up
    text = re.sub(r"\s+", " ", text)    # collapse multiple spaces
    return text


def fact_found_in_answer(fact: str, answer: str) -> bool:
    """
    Check if a key fact appears in the answer, with fuzzy matching
    for hyphens and spacing variants.
    """
    return normalize_text(fact) in normalize_text(answer)


def score_answer_quality(answer: str, key_facts: list[str]) -> float:
    """
    What fraction of expected key facts appear in the answer?

    Returns a float from 0.0 to 1.0. A score of 1.0 means all
    key facts were mentioned. Uses normalized matching so that
    'cleanup', 'clean-up', and 'clean up' all count as matches.

    For more sophisticated scoring, you'd use an LLM-as-judge
    (a future enhancement listed in Phase 6).
    """
    if not key_facts:
        return 1.0  # No key facts specified = auto-pass

    found = [f for f in key_facts if fact_found_in_answer(f, answer)]
    return len(found) / len(key_facts)


# ─────────────────────────────────────────────────────────────────
# Eval runner
# ─────────────────────────────────────────────────────────────────

def run_single_eval(question: dict, verbose: bool = False) -> dict:
    """Run a single eval question and return scored results."""
    q_text = question["question"]

    # Run the full pipeline
    try:
        result = answer_question(q_text)
        answer = result["answer"]
        sources = result["sources"]
        query_type = result["query_type"]
        error = None
    except Exception as e:
        answer = ""
        sources = []
        query_type = "error"
        error = str(e)

    # Score
    routing_correct = score_routing(question["expected_type"], query_type)
    retrieval_hit = score_retrieval(sources, question.get("expected_source_keywords", []))
    answer_quality = score_answer_quality(answer, question.get("key_facts", []))

    # Find which key facts were found/missed
    key_facts = question.get("key_facts", [])
    facts_found = [f for f in key_facts if fact_found_in_answer(f, answer)]
    facts_missed = [f for f in key_facts if not fact_found_in_answer(f, answer)]

    result_row = {
        "question": q_text,
        "expected_type": question["expected_type"],
        "actual_type": query_type,
        "routing_correct": routing_correct,
        "retrieval_hit": retrieval_hit,
        "answer_quality": round(answer_quality, 2),
        "facts_found": "; ".join(facts_found),
        "facts_missed": "; ".join(facts_missed),
        "num_sources": len([s for s in sources if s["type"] != "meta"]),
        "error": error or "",
    }

    if verbose:
        print(f"\n    Answer: {answer[:200]}...")
        if facts_missed:
            print(f"    Missed facts: {facts_missed}")

    return result_row


def run_all_evals(
    questions: list[dict],
    verbose: bool = False,
    filter_type: str | None = None,
    single_index: int | None = None,
) -> list[dict]:
    """Run all eval questions and return results."""
    results = []

    # Apply filters
    if single_index is not None:
        if 0 <= single_index < len(questions):
            questions = [questions[single_index]]
        else:
            print(f"Error: Question index {single_index} out of range (0–{len(questions)-1})")
            sys.exit(1)

    if filter_type:
        questions = [q for q in questions if q["expected_type"] == filter_type]
        if not questions:
            print(f"Error: No questions found for type '{filter_type}'")
            sys.exit(1)

    total = len(questions)
    for i, q in enumerate(questions):
        q_text = q["question"]
        q_type = q["expected_type"]
        print(f"  [{i+1}/{total}] ({q_type}) \"{q_text}\"")

        result = run_single_eval(q, verbose=verbose)

        # Print inline result
        r_icon = "✅" if result["routing_correct"] else "❌"
        s_icon = "✅" if result["retrieval_hit"] else "❌"
        a_score = result["answer_quality"]
        a_icon = "✅" if a_score >= 0.75 else "⚠️" if a_score >= 0.5 else "❌"
        print(f"         Route: {r_icon}  Retrieval: {s_icon}  Quality: {a_icon} ({a_score:.0%})")

        if result["error"]:
            print(f"         ⚠️ Error: {result['error']}")

        results.append(result)
        time.sleep(RATE_LIMIT_DELAY)

    return results


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    """Print aggregate metrics and per-type breakdown."""
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    # Aggregate metrics
    routing_acc = sum(r["routing_correct"] for r in results) / total
    retrieval_prec = sum(r["retrieval_hit"] for r in results) / total
    avg_quality = sum(r["answer_quality"] for r in results) / total

    print(f"\n{'=' * 60}")
    print(f"  EVAL RESULTS — {total} questions")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")
    print(f"  Routing accuracy:     {routing_acc:>6.0%}  {'✅' if routing_acc >= 0.85 else '⚠️'} (target: ≥85%)")
    print(f"  Retrieval precision:  {retrieval_prec:>6.0%}  {'✅' if retrieval_prec >= 0.80 else '⚠️'} (target: ≥80%)")
    print(f"  Answer quality:       {avg_quality:>6.0%}  {'✅' if avg_quality >= 0.75 else '⚠️'} (target: ≥75%)")

    # Per-type breakdown
    print(f"\n  {'Type':<20} {'Route':>6} {'Retrieval':>10} {'Quality':>8}  {'N':>3}")
    print(f"  {'─' * 52}")

    types = sorted(set(r["expected_type"] for r in results))
    for t in types:
        t_results = [r for r in results if r["expected_type"] == t]
        n = len(t_results)
        t_route = sum(r["routing_correct"] for r in t_results) / n
        t_retr = sum(r["retrieval_hit"] for r in t_results) / n
        t_qual = sum(r["answer_quality"] for r in t_results) / n
        print(f"  {t:<20} {t_route:>5.0%} {t_retr:>9.0%} {t_qual:>7.0%}  {n:>3}")

    # Failures detail
    failures = [r for r in results if not r["routing_correct"] or not r["retrieval_hit"] or r["answer_quality"] < 0.5]
    if failures:
        print(f"\n  ── Issues to Investigate ──")
        for r in failures:
            issues = []
            if not r["routing_correct"]:
                issues.append(f"routed as {r['actual_type']} (expected {r['expected_type']})")
            if not r["retrieval_hit"]:
                issues.append("correct source not retrieved")
            if r["answer_quality"] < 0.5:
                missed = r.get("facts_missed", "")
                issues.append(f"low quality ({r['answer_quality']:.0%}), missed: {missed}")
            print(f"  ⚠️  \"{r['question']}\"")
            for issue in issues:
                print(f"      → {issue}")

    # Improvement suggestions
    print(f"\n  ── Improvement Suggestions ──")
    if routing_acc < 0.85:
        print("  📌 Routing: Add more keyword rules to router.py for misclassified questions.")
    if retrieval_prec < 0.80:
        print("  📌 Retrieval: Try smaller chunks (300 tokens) or add expansion pre-filters.")
    if avg_quality < 0.75:
        print("  📌 Quality: Check if the right context IS retrieved (retrieval issue) or")
        print("              if the LLM is ignoring context (prompt issue). Different fixes needed.")
    if routing_acc >= 0.85 and retrieval_prec >= 0.80 and avg_quality >= 0.75:
        print("  🎉 All metrics meet target benchmarks! Phase 5 is complete.")


def save_results(results: list[dict]) -> None:
    """Save detailed results to CSV."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  💾 Detailed results saved to {RESULTS_PATH}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run DominionSage evaluation suite.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full answers and missed facts.")
    parser.add_argument("--type", "-t", type=str, default=None,
                        choices=["card_lookup", "filtered_search", "rules_question", "strategy_combo"],
                        help="Only run evals for a specific query type.")
    parser.add_argument("--question", "-q", type=int, default=None,
                        help="Run a single question by index (0-based).")
    args = parser.parse_args()

    print("=" * 60)
    print("  DominionSage — Evaluation Suite")
    print("=" * 60)

    # Load questions
    if not QUESTIONS_PATH.exists():
        print(f"Error: {QUESTIONS_PATH} not found.")
        sys.exit(1)

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"  Loaded {len(questions)} eval questions from {QUESTIONS_PATH}")

    if args.type:
        print(f"  Filtering to type: {args.type}")
    if args.question is not None:
        print(f"  Running single question: #{args.question}")

    print()

    # Run evals
    results = run_all_evals(
        questions,
        verbose=args.verbose,
        filter_type=args.type,
        single_index=args.question,
    )

    # Report
    print_summary(results)

    # Save
    save_results(results)


if __name__ == "__main__":
    main()
