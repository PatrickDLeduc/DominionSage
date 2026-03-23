import random
from simulation.runner import run_simulation
from simulation.cards import CARD_DEFS

# Get all 26 base set kingdom cards
base_set = [name for name, d in CARD_DEFS.items() if name not in 
            ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]]

print(f"Total base set cards available: {len(base_set)}")

# Pick 10 random
kingdom = random.sample(base_set, 10)
print(f"Simulating Kingdom: {', '.join(kingdom)}")

stats = run_simulation(kingdom, n_games=100)
print(f"\nWin rates: {stats['win_rates']}")
print(f"Avg VP: {stats['avg_vp']}")
print(f"Avg turns: {stats['avg_turns']}")

print("\nEngine Buys:")
for card, avg in stats['buy_frequency']['Engine'].items():
    if avg >= 0.5:
        print(f"  {card}: {avg}")
