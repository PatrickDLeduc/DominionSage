from simulation.runner import run_simulation
from simulation.bots import AVAILABLE_BOTS

kingdom = ["Chapel", "Witch", "Militia", "Village", "Market", 
           "Festival", "Bandit", "Library", "Sentry", "Gardens"]

print("Testing 3 Player Game: Attacker vs Big Money vs Rusher...")
stats3 = run_simulation(kingdom, n_games=10, 
                        bot_classes=[AVAILABLE_BOTS["Attacker"], 
                                     AVAILABLE_BOTS["Big Money"],
                                     AVAILABLE_BOTS["Rusher"]])
print("Winner:", max(stats3["wins"], key=stats3["wins"].get))
print("Turns:", stats3["avg_turns"])


print("\nTesting 4 Player Game: Engine vs TrashBot vs Attacker vs Big Money...")
stats4 = run_simulation(kingdom, n_games=10, 
                        bot_classes=[AVAILABLE_BOTS["Engine"], 
                                     AVAILABLE_BOTS["TrashBot"],
                                     AVAILABLE_BOTS["Attacker"],
                                     AVAILABLE_BOTS["Big Money"]])
print("Winner:", max(stats4["wins"], key=stats4["wins"].get))
print("Turns:", stats4["avg_turns"])

print("\nAll multi-player bots ran without crashing!")
