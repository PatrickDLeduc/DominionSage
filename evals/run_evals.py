"""
run_evals.py — DominionSage Evaluation Suite (Phase 5)

Runs all 20 eval questions through the full pipeline and scores them
on three dimensions:

  1. Routing accuracy  — Did the router classify the query correctly?
  2. Retrieval precision — Did the correct source appear in the results?
  3. Answer quality — Does the generated answer contain the key facts?

Two scoring modes:
  - Default: keyword-based scoring (fast, free, catches obvious failures)
  - --judge: LLM-as-judge scoring (GPT-4o grades each answer semantically)

The keyword scorer is like a scantron machine — it checks if the right
bubbles are filled in. The LLM judge is like a teaching assistant who
reads the essay, understands the meaning, and catches wrong answers
that happen to contain the right keywords.

Usage:
  python evals/run_evals.py                  # keyword scoring only
  python evals/run_evals.py --judge          # add LLM-as-judge scoring
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

    This is the "fast and free" scorer. For deeper evaluation,
    use --judge to enable the LLM-as-judge scorer.
    """
    if not key_facts:
        return 1.0  # No key facts specified = auto-pass

    found = [f for f in key_facts if fact_found_in_answer(f, answer)]
    return len(found) / len(key_facts)


# ─────────────────────────────────────────────────────────────────
# LLM-as-Judge scoring
# ─────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator for a Dominion card game AI assistant.

Your job is to grade the assistant's answer on three criteria. Be strict but fair.

CRITERIA:

1. FACTUAL ACCURACY (1-5):
   Is the information correct? Check for:
   - Wrong card costs (e.g., saying Chapel costs 3 when it costs 2)
   - Wrong card effects (e.g., saying Smithy draws 2 cards when it draws 3)
   - Wrong expansion attributions
   - Incorrect rule interpretations
   - Cards listed under wrong categories
   Score 5 = all facts correct. Score 1 = major factual errors.

2. COMPLETENESS (1-5):
   Does the answer cover what was asked?
   - For card lookups: does it mention the key mechanics?
   - For filtered searches: are the right cards included (and wrong ones excluded)?
   - For rules questions: is the ruling complete and precise?
   - For strategy questions: does it explain WHY, not just WHAT?
   Score 5 = fully complete. Score 1 = misses the main point.

3. SOURCE QUALITY (1-5):
   Are sources cited correctly?
   - Are rulebook page numbers plausible for the content cited?
   - Does the answer distinguish between card database facts and rulebook rules?
   - Does it avoid making claims not supported by the provided context?
   Score 5 = excellent sourcing. Score 1 = fabricated or misleading sources.

RESPOND IN EXACTLY THIS JSON FORMAT (no other text):
{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "source_quality": <1-5>,
  "overall": <1-5>,
  "errors": ["list any specific factual errors found"],
  "reasoning": "brief explanation of your scores"
}"""


_judge_client = None

def _get_judge_client():
    """Lazy-initialize the OpenAI client for the judge."""
    global _judge_client
    if _judge_client is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from openai import OpenAI
        _judge_client = OpenAI()
    return _judge_client


def llm_judge_score(
    question: str,
    answer: str,
    expected_type: str,
    key_facts: list[str],
) -> dict:
    """
    Use GPT-4o to grade an answer on accuracy, completeness, and sourcing.

    This is the "teaching assistant" scorer — it reads the answer,
    understands the meaning, and catches errors that keyword matching
    can't detect.

    Returns a dict with scores (1-5), errors list, and reasoning.
    Cost: ~$0.01-0.03 per call.
    """
    client = _get_judge_client()

    # Build the grading context
    facts_str = ", ".join(key_facts) if key_facts else "(no specific facts required)"

    user_message = f"""QUESTION TYPE: {expected_type}

USER QUESTION: {question}

ASSISTANT'S ANSWER:
{answer}

EXPECTED KEY FACTS: {facts_str}

Grade this answer according to the criteria. Remember to respond ONLY in the JSON format specified."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON (handle markdown code fences if present)
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(result_text)

        # Validate expected fields
        required = ["accuracy", "completeness", "source_quality", "overall", "errors", "reasoning"]
        for field in required:
            if field not in scores:
                scores[field] = 3 if field != "errors" else []
                if field == "reasoning":
                    scores[field] = "Parse error — field missing"

        return scores

    except json.JSONDecodeError:
        return {
            "accuracy": 3,
            "completeness": 3,
            "source_quality": 3,
            "overall": 3,
            "errors": ["Judge response was not valid JSON"],
            "reasoning": f"Raw response: {result_text[:200]}",
        }
    except Exception as e:
        return {
            "accuracy": 0,
            "completeness": 0,
            "source_quality": 0,
            "overall": 0,
            "errors": [f"Judge API error: {str(e)}"],
            "reasoning": "Failed to call judge API.",
        }


# ─────────────────────────────────────────────────────────────────
# Eval runner
# ─────────────────────────────────────────────────────────────────

def run_single_eval(question: dict, verbose: bool = False, use_judge: bool = False) -> dict:
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

    # ── Keyword scoring (always runs) ──
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

    # ── LLM Judge scoring (only with --judge flag) ──
    if use_judge and not error:
        judge_scores = llm_judge_score(
            question=q_text,
            answer=answer,
            expected_type=question["expected_type"],
            key_facts=key_facts,
        )
        result_row["judge_accuracy"] = judge_scores["accuracy"]
        result_row["judge_completeness"] = judge_scores["completeness"]
        result_row["judge_source_quality"] = judge_scores["source_quality"]
        result_row["judge_overall"] = judge_scores["overall"]
        result_row["judge_errors"] = "; ".join(judge_scores.get("errors", []))
        result_row["judge_reasoning"] = judge_scores.get("reasoning", "")

    if verbose:
        print(f"\n    Answer: {answer[:200]}...")
        if facts_missed:
            print(f"    Missed facts: {facts_missed}")
        if use_judge and "judge_overall" in result_row:
            print(f"    Judge: {result_row['judge_overall']}/5 — {result_row.get('judge_reasoning', '')[:100]}")
            if result_row.get("judge_errors"):
                print(f"    Judge errors: {result_row['judge_errors']}")

    return result_row


def run_all_evals(
    questions: list[dict],
    verbose: bool = False,
    filter_type: str | None = None,
    single_index: int | None = None,
    use_judge: bool = False,
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

        result = run_single_eval(q, verbose=verbose, use_judge=use_judge)

        # Print inline result
        r_icon = "✅" if result["routing_correct"] else "❌"
        s_icon = "✅" if result["retrieval_hit"] else "❌"
        a_score = result["answer_quality"]
        a_icon = "✅" if a_score >= 0.75 else "⚠️" if a_score >= 0.5 else "❌"

        line = f"         Route: {r_icon}  Retrieval: {s_icon}  Quality: {a_icon} ({a_score:.0%})"

        # Add judge score if available
        if "judge_overall" in result:
            j = result["judge_overall"]
            j_icon = "✅" if j >= 4 else "⚠️" if j >= 3 else "❌"
            line += f"  Judge: {j_icon} ({j}/5)"

        print(line)

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

    has_judge = "judge_overall" in results[0]

    print(f"\n{'=' * 60}")
    print(f"  EVAL RESULTS — {total} questions")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")
    print(f"  Routing accuracy:     {routing_acc:>6.0%}  {'✅' if routing_acc >= 0.85 else '⚠️'} (target: ≥85%)")
    print(f"  Retrieval precision:  {retrieval_prec:>6.0%}  {'✅' if retrieval_prec >= 0.80 else '⚠️'} (target: ≥80%)")
    print(f"  Answer quality (kw):  {avg_quality:>6.0%}  {'✅' if avg_quality >= 0.75 else '⚠️'} (target: ≥75%)")

    if has_judge:
        avg_accuracy = sum(r.get("judge_accuracy", 0) for r in results) / total
        avg_complete = sum(r.get("judge_completeness", 0) for r in results) / total
        avg_sources = sum(r.get("judge_source_quality", 0) for r in results) / total
        avg_overall = sum(r.get("judge_overall", 0) for r in results) / total

        print(f"\n  ── LLM Judge Scores (GPT-4o, 1-5 scale) ──")
        print(f"  Factual accuracy:     {avg_accuracy:>5.1f}  {'✅' if avg_accuracy >= 4 else '⚠️' if avg_accuracy >= 3 else '❌'}")
        print(f"  Completeness:         {avg_complete:>5.1f}  {'✅' if avg_complete >= 4 else '⚠️' if avg_complete >= 3 else '❌'}")
        print(f"  Source quality:        {avg_sources:>5.1f}  {'✅' if avg_sources >= 4 else '⚠️' if avg_sources >= 3 else '❌'}")
        print(f"  Overall:              {avg_overall:>5.1f}  {'✅' if avg_overall >= 4 else '⚠️' if avg_overall >= 3 else '❌'}")

    # Per-type breakdown
    header = f"  {'Type':<20} {'Route':>6} {'Retrieval':>10} {'Kw Qual':>8}"
    if has_judge:
        header += f"  {'Judge':>6}"
    header += f"  {'N':>3}"
    print(f"\n{header}")
    print(f"  {'─' * (58 if has_judge else 50)}")

    types = sorted(set(r["expected_type"] for r in results))
    for t in types:
        t_results = [r for r in results if r["expected_type"] == t]
        n = len(t_results)
        t_route = sum(r["routing_correct"] for r in t_results) / n
        t_retr = sum(r["retrieval_hit"] for r in t_results) / n
        t_qual = sum(r["answer_quality"] for r in t_results) / n
        line = f"  {t:<20} {t_route:>5.0%} {t_retr:>9.0%} {t_qual:>7.0%}"
        if has_judge:
            t_judge = sum(r.get("judge_overall", 0) for r in t_results) / n
            line += f"  {t_judge:>5.1f}"
        line += f"  {n:>3}"
        print(line)

    # Failures detail
    failures = [r for r in results if not r["routing_correct"] or not r["retrieval_hit"] or r["answer_quality"] < 0.5]

    # Add judge failures (scored 2 or below)
    if has_judge:
        judge_failures = [r for r in results if r.get("judge_overall", 5) <= 2 and r not in failures]
        failures.extend(judge_failures)

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
                issues.append(f"low keyword quality ({r['answer_quality']:.0%}), missed: {missed}")
            if has_judge and r.get("judge_overall", 5) <= 2:
                issues.append(f"judge score: {r['judge_overall']}/5")
                if r.get("judge_errors"):
                    issues.append(f"judge found errors: {r['judge_errors']}")
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
    if has_judge:
        avg_overall = sum(r.get("judge_overall", 0) for r in results) / total
        if avg_overall < 3.5:
            print("  📌 Judge: Low overall scores suggest factual errors in answers.")
            print("              Check the judge_errors column in results.csv for specifics.")
        avg_sources = sum(r.get("judge_source_quality", 0) for r in results) / total
        if avg_sources < 3.5:
            print("  📌 Sources: Judge flagged poor source citations. Consider adding")
            print("              chunk-level relevance thresholds or post-generation verification.")
    if routing_acc >= 0.85 and retrieval_prec >= 0.80 and avg_quality >= 0.75:
        if not has_judge or avg_overall >= 3.5:
            print("  🎉 All metrics meet target benchmarks!")


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
    parser.add_argument("--judge", "-j", action="store_true",
                        help="Enable LLM-as-judge scoring (GPT-4o, ~$0.50 for 20 questions).")
    parser.add_argument("--type", "-t", type=str, default=None,
                        choices=["card_lookup", "filtered_search", "rules_question", "strategy_combo"],
                        help="Only run evals for a specific query type.")
    parser.add_argument("--question", "-q", type=int, default=None,
                        help="Run a single question by index (0-based).")
    args = parser.parse_args()

    print("=" * 60)
    print("  DominionSage — Evaluation Suite")
    if args.judge:
        print("  (LLM-as-Judge enabled — GPT-4o will grade each answer)")
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
        use_judge=args.judge,
    )

    # Report
    print_summary(results)

    # Save
    save_results(results)


if __name__ == "__main__":
    main()
