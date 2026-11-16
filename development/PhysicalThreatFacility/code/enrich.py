#!/usr/bin/env python3
"""
ENRICHMENT - only when the message itself has a link. Find the links (urlextract), open each one in a real browser
(Playwright, headless Chromium), take all the text on the page and add it to the message:
    message + "\n[linked page] " + page text        (one block per link that opened)
Nothing else: no lookups, no picking fields. A link that does not open adds nothing.

    from enrich import Enricher
    with Enricher() as e: text, links = e.enrich("Fire downtown https://example.com/story")
    python enrich.py "some message with a link"

Setup once:  pip install urlextract playwright && playwright install chromium   (or have Google Chrome installed)
"""
import re, sys
from urlextract import URLExtract

class Enricher:
    def __init__(self, timeout_ms=30000, render_wait_ms=4000):
        self.x, self.timeout, self.wait = URLExtract(), timeout_ms, render_wait_ms

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self.p = sync_playwright().start()
        try: self.browser = self.p.chromium.launch(headless=True)                        # Playwright's own Chromium
        except Exception: self.browser = self.p.chromium.launch(channel="chrome", headless=True)   # else the installed Google Chrome
        return self

    def __exit__(self, *a):
        self.browser.close(); self.p.stop()

    def scrape(self, url):
        ctx = self.browser.new_context(locale="en-US"); page = ctx.new_page()
        try:
            r = page.goto(url if "://" in url else "https://" + url, wait_until="domcontentloaded", timeout=self.timeout)
            page.wait_for_timeout(self.wait)                                  # let JavaScript render the page
            text = re.sub(r"\s+", " ", page.inner_text("body")).strip(); status = r.status if r else None
            ok = bool(status and status < 400 and len(text) > 200)
            return {"url": url, "final_url": page.url, "status": status, "ok": ok, "chars": len(text), "text": text if ok else "",
                    "error": "" if ok else f"HTTP {status}" if status and status >= 400 else f"only {len(text)} characters on the page"}
        except Exception as e:
            return {"url": url, "final_url": page.url, "status": None, "ok": False, "chars": 0, "text": "", "error": str(e).splitlines()[0][:200]}
        finally:
            ctx.close()

    def enrich(self, text):
        """-> (text for the models, per-link results). No link -> the message unchanged."""
        links = [self.scrape(u) for u in self.x.find_urls(text or "", only_unique=True)]
        return text + "".join("\n[linked page] " + l["text"] for l in links if l["ok"]), links

if __name__ == "__main__":
    with Enricher() as e:
        for t in sys.argv[1:]:
            out, links = e.enrich(t)
            for l in links: print(("✓" if l["ok"] else "✗"), l["url"], l["chars"], l["error"])
            print(out[:500])
