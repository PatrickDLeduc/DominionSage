# test_wiki.py
import requests

API = "https://wiki.dominionstrategy.com/api.php"

test_names = ["Chapel", "Village", "Throne Room", "Throne_Room", "Alms", "Militia"]

for name in test_names:
    resp = requests.get(API, params={"action": "query", "titles": name, "format": "json"})
    pages = resp.json().get("query", {}).get("pages", {})
    page_ids = list(pages.keys())
    exists = "-1" not in page_ids
    print(f"  {name}: {'EXISTS' if exists else 'NOT FOUND'} (ids: {page_ids})")
    