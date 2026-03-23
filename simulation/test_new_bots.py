from simulation.runner import run_simulation
from simulation.bots import AVAILABLE_BOTS

# Use a kingdom that lets everyone shine
kingdom = ["Chapel", "Witch", "Militia", "Village", "Market", 
           "Festival", "Bandit", "Library", "Sentry", "Gardens"]

print("Testing Attacker vs Big Money...")
stats1 = run_simulation(kingdom, n_games=10, 
                        bot1_cls=AVAILABLE_BOTS["Attacker"], 
                        bot2_cls=AVAILABLE_BOTS["Big Money"])
print("Winner:", max(stats1["wins"], key=stats1["wins"].get))

print("\nTesting TrashBot vs Engine...")
stats2 = run_simulation(kingdom, n_games=10, 
                        bot1_cls=AVAILABLE_BOTS["TrashBot"], 
                        bot2_cls=AVAILABLE_BOTS["Engine"])
print("Winner:", max(stats2["wins"], key=stats2["wins"].get))

print("\nTesting Rusher vs Attacker...")
stats3 = run_simulation(kingdom, n_games=10, 
                        bot1_cls=AVAILABLE_BOTS["Rusher"], 
                        bot2_cls=AVAILABLE_BOTS["Attacker"])
print("Winner:", max(stats3["wins"], key=stats3["wins"].get))

print("\nAll bots ran without crashing!")
