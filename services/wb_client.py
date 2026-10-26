"""Wildberries: search and product characteristics, through the Ozon client's browser.

Plain requests to search.wb.ru are 403 (2026-10-26). The site's own search call is
read off the opened search page; a card's characteristics are the static
card.json on the basket-NN.wbbasket.ru CDN, whose NN the site computes from the
card's number -- the browser asks the first 40 hosts at once and keeps the one
that answers (10 cards in ~0.3 s).
"""
from __future__ import annotations

import logging
import urllib.parse

import ozon_client as _oz

logger = logging.getLogger("assistant.wb")

SEARCH_URL = "https://www.wildberries.ru/catalog/0/search.aspx?search="
CARDS_JS = """async ids => { const out = {};
  await Promise.all(ids.map(async id => {
    const vol = Math.floor(id / 100000), part = Math.floor(id / 1000);
    const hosts = Array.from({length: 40}, (_, i) => String(i + 1).padStart(2, '0'));
    try { out[id] = await Promise.any(hosts.map(h =>
        fetch(`https://basket-${h}.wbbasket.ru/vol${vol}/part${part}/${id}/info/ru/card.json`)
          .then(r => r.ok ? r.json() : Promise.reject(r.status)))); }
    catch (e) { out[id] = null; } }));
  return out; }"""


def parse_search(data: dict) -> list:
    out = []
    for p in (data or {}).get("products") or []:
        price = next((s["price"].get("product") for s in p.get("sizes") or []
                      if isinstance(s.get("price"), dict) and s["price"].get("product")), None)
        if not (p.get("id") and price):
            continue
        out.append({"nm": p["id"], "name": f"{p.get('brand') or ''} {p.get('name') or ''}".strip(),
                    "price": int(price) // 100, "seller": p.get("supplier"),
                    "rating": p.get("reviewRating") or p.get("rating"), "reviews": p.get("feedbacks"),
                    "url": f"https://www.wildberries.ru/catalog/{p['id']}/detail.aspx"})
    return out


def parse_card(card) -> dict:
    if not isinstance(card, dict):
        return {}
    ch = {o["name"]: str(o["value"]) for o in card.get("options") or []
          if o.get("name") and o.get("value") and "упаковки" not in o["name"]}
    if card.get("vendor_code"):
        ch["Артикул производителя"] = card["vendor_code"]
    return ch


def search(query: str, cards: int = 30) -> list:
    """Search results, the first `cards` of them with their characteristics."""
    def run(page):
        got = []

        def on(r):
            if not got and r.status == 200 and "/search?" in r.url and "u-search" in r.url:
                try:
                    got.append(r.json())
                except Exception:
                    logger.debug("WB search answer was not JSON", exc_info=True)
        page.on("response", on)
        try:
            page.goto(SEARCH_URL + urllib.parse.quote(query), wait_until="domcontentloaded", timeout=60000)
            for _ in range(30):
                if got:
                    break
                page.wait_for_timeout(300)
        finally:
            page.remove_listener("response", on)
        items = parse_search(got[0] if got else {})
        if items and cards:
            found = page.evaluate(CARDS_JS, [x["nm"] for x in items[:cards]])
            for x in items[:cards]:
                x["characteristics"] = parse_card(found.get(str(x["nm"])))
        return items
    return _oz.fetch_page(run)
