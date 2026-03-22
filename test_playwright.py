# test_playwright.py
# First install: pip install playwright
# Then run: playwright install chromium

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://wiki.dominionstrategy.com/api.php?action=parse&page=Chapel&prop=wikitext&format=json")
    page.wait_for_timeout(5000)  # Wait for Anubis challenge to complete
    content = page.content()
    print(content[:500])
    browser.close()