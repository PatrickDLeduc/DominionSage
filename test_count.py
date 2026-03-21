# test_count.py
from retrieval.card_lookup import filtered_search

cards = filtered_search({"type": "Action", "max_cost": 4})
print(f"Total: {len(cards)} cards")
for c in cards:
    print(f"  {c['name']}")