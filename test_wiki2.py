# test_wiki2.py
import requests

API = "https://wiki.dominionstrategy.com/api.php"

resp = requests.get(API, params={"action": "query", "titles": "Chapel", "format": "json"})
print(f"Status: {resp.status_code}")
print(f"Headers: {dict(resp.headers)}")
print(f"Body (first 500 chars): {resp.text[:500]}")