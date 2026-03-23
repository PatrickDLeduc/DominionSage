"""
simulation/runner.py — Run N-game simulations and analyze results with GPT.

Orchestrates bot-vs-bot games, collects statistics, and sends
a summary to GPT-4o-mini for strategy insight discovery.
"""

from __future__ import annotations

import time
from collections import Counter

from simulation.engine import run_game
from simulation.bots import BigMoneyBot, EngineBot


# ─────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────

def run_simulation(kingdom: list[str], n_games: int = 200,
                   bot1_cls=BigMoneyBot, bot2_cls=EngineBot) -> dict:
    """
    Run n_games of bot1 vs bot2 on the given kingdom and return stats.

    Returns:
      - bot_names: [str, str]
      - wins: {name: int}
      - ties: int
      - avg_vp: {name: float}
      - avg_turns: float
      - buy_frequency: {name: {card: avg_count}}
      - sample_log: log from the last game (for GPT context)
    """
    bot1 = bot1_cls()
    bot2 = bot2_cls()

    wins = {bot1.name: 0, bot2.name: 0}
    ties = 0
    total_vp = {bot1.name: 0, bot2.name: 0}
    total_turns = 0
    all_buy_counts = {bot1.name: Counter(), bot2.name: Counter()}
    sample_log = []

    for i in range(n_games):
        # Alternate who goes first to reduce first-player advantage
        if i % 2 == 0:
            bots = [bot1, bot2]
            names = [bot1.name, bot2.name]
        else:
            bots = [bot2, bot1]
            names = [bot2.name, bot1.name]

        result = run_game(kingdom, bots, player_names=names)

        # Track wins
        if result["winner"] == "Tie":
            ties += 1
        else:
            wins[result["winner"]] = wins.get(result["winner"], 0) + 1

        # Track VP
        for name, vp in result["vp"].items():
            total_vp[name] += vp

        # Track turns
        total_turns += result["turns"]

        # Track buy frequency
        for name, buys in result["buy_counts"].items():
            for card, count in buys.items():
                all_buy_counts[name][card] += count

        # Keep last game log as sample
        sample_log = result["log"]

    # Compute averages
    avg_vp = {name: round(vp / n_games, 1) for name, vp in total_vp.items()}
    avg_turns = round(total_turns / n_games, 1)
    buy_frequency = {
        name: {card: round(count / n_games, 2)
               for card, count in sorted(counts.items(),
                                         key=lambda x: -x[1])}
        for name, counts in all_buy_counts.items()
    }

    return {
        "bot_names": [bot1.name, bot2.name],
        "n_games": n_games,
        "kingdom": kingdom,
        "wins": wins,
        "ties": ties,
        "win_rates": {name: round(w / n_games * 100, 1) for name, w in wins.items()},
        "avg_vp": avg_vp,
        "avg_turns": avg_turns,
        "buy_frequency": buy_frequency,
        "sample_log": sample_log,
    }


# ─────────────────────────────────────────────────────────────────
# GPT analysis of simulation results
# ─────────────────────────────────────────────────────────────────

def format_stats_for_llm(stats: dict) -> str:
    """Format simulation stats into a readable prompt for GPT analysis."""
    lines = [
        f"## Dominion Simulation Results",
        f"**Kingdom**: {', '.join(stats['kingdom'])}",
        f"**Games played**: {stats['n_games']}",
        f"**Bots**: {stats['bot_names'][0]} vs {stats['bot_names'][1]}",
        "",
        f"### Win Rates",
    ]

    for name in stats["bot_names"]:
        lines.append(f"- **{name}**: {stats['wins'][name]} wins "
                     f"({stats['win_rates'][name]}%)")
    lines.append(f"- **Ties**: {stats['ties']}")

    lines.append(f"\n### Average VP per game")
    for name, vp in stats["avg_vp"].items():
        lines.append(f"- {name}: {vp} VP")

    lines.append(f"\n**Average game length**: {stats['avg_turns']} turns")

    lines.append(f"\n### Average Cards Purchased per Game")
    for name, buys in stats["buy_frequency"].items():
        lines.append(f"\n**{name}**:")
        for card, avg in buys.items():
            if avg >= 0.1:
                lines.append(f"  - {card}: {avg}")

    # Include a sample game log (truncated)
    log_sample = stats["sample_log"][:60]
    lines.append(f"\n### Sample Game Log (first 60 lines)")
    lines.extend(log_sample)

    return "\n".join(lines)


def analyze_simulation(stats: dict) -> str:
    """
    Send simulation stats to GPT-4o-mini for strategy analysis.

    Returns the LLM's strategy insights as a string.
    """
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    stats_text = format_stats_for_llm(stats)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Dominion strategy analyst. You have been given "
                    "the results of bot-vs-bot simulations on a specific kingdom. "
                    "Analyze the data and provide strategic insights:\n\n"
                    "1. **Why the winner wins** — what cards/strategy make the difference\n"
                    "2. **Key card interactions** — combos that matter in this kingdom\n"
                    "3. **Optimal opening** — what to buy on turns 1-2 based on the data\n"
                    "4. **Strategic recommendations** — advice for a human player\n"
                    "5. **Surprising findings** — anything unexpected in the data\n\n"
                    "Be specific and reference the numbers. Keep it concise but insightful."
                ),
            },
            {
                "role": "user",
                "content": stats_text,
            },
        ],
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kingdom = ["Cellar", "Market", "Merchant", "Militia", "Mine",
               "Moat", "Remodel", "Smithy", "Village", "Workshop"]

    print("Running 200 simulations (Big Money vs Engine)...")
    start = time.time()
    stats = run_simulation(kingdom, n_games=200)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"\nWin rates:")
    for name in stats["bot_names"]:
        print(f"  {name}: {stats['wins'][name]} wins ({stats['win_rates'][name]}%)")
    print(f"  Ties: {stats['ties']}")
    print(f"\nAvg VP: {stats['avg_vp']}")
    print(f"Avg turns: {stats['avg_turns']}")
    print(f"\nBuy frequency:")
    for name, buys in stats["buy_frequency"].items():
        print(f"  {name}: {dict(list(buys.items())[:8])}")
