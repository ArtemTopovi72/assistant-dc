"""Ozon answers as tappable cards.

A text answer with five links is hard to follow: which link was the 512 ₽
one? After the answer goes out, every product it links to gets its own
message -- the product photo, name, price, delivery date and an "open on
Ozon" button -- in the order the answer mentions them. Only products the
answer actually links to are sent: the tool may have seen 60.
"""
from __future__ import annotations

import html
import logging
import re

logger = logging.getLogger("assistant.tg.cards")

MAX_CARDS = 8


def _rub(n) -> str:
    return f"{n:,}".replace(",", " ") + " ₽" if isinstance(n, int) else ""


def _reviews_ru(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "отзывов"
    if n % 10 == 1 and n % 100 != 11:
        return "отзыв"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "отзыва"
    return "отзывов"


# A line of the text reply that only restates a card: it links to a carded
# product. 2026-09-24, user: the answer listed five products as
# "…доставка сегодня: ссылка (https://www.ozon.ru/product/…)" and then the
# same five came again as cards -- the list is noise once the cards exist.


def _heading(line: str) -> bool:
    return line.rstrip().rstrip("*").rstrip().endswith(":")


def condense(reply: str, products: list) -> str:
    """The reply without the lines the cards repeat. Keeps the lead-in and
    anything the model said about its choice; empty lead-in gets a short one."""
    picked = pick(reply, products)
    if not picked:
        return reply
    skus = [str(p["sku"]) for p in picked]
    keep = []
    for line in (reply or "").splitlines():
        if any(re.search(r"ozon\.ru/\S*?(?<!\d)" + re.escape(s) + r"(?!\d)", line) for s in skus):
            last = next((j for j in range(len(keep) - 1, -1, -1) if keep[j].strip()), None)
            if last is not None and _heading(keep[last]):
                # Its section went into the cards: a sub-heading («Мелочёвка и
                # запчасти:») would stand over nothing (live 2026-09-27 20:25,
                # two bare headings); the first lead-in points at the cards.
                if any(k.strip() for k in keep[:last]):
                    del keep[last]
                else:
                    keep[last] = re.sub(r":(\**)\s*$", r"\1 👇", keep[last].rstrip())
            continue
        keep.append(line)
    first = next((j for j, k in enumerate(keep) if k.strip()), None)
    if first is not None and _heading(keep[first]):
        keep[first] = re.sub(r":(\**)\s*$", r"\1 👇", keep[first].rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()
    return text or f"Нашёл на Ozon — {len(picked)} 👇"


def _photo_bytes(url: str) -> bytes | None:
    """Fetch the photo ourselves. Handing Telegram the URL made the LOCAL Bot
    API server fetch it through this machine's VPN, which Ozon's CDN answers
    only sometimes -- a card arrived text-only with no error anywhere."""
    import requests
    try:
        import ozon_client
        proxy = getattr(ozon_client, "OZON_PROXY", "") or None
    except Exception:
        proxy = None
    for proxies in (({"https": proxy, "http": proxy} if proxy else None), None):
        try:
            r = requests.get(url, timeout=20, proxies=proxies,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image/"):
                return r.content
        except Exception as exc:
            logger.info("card photo fetch %s failed: %s", url, exc)
        if not proxy:
            break
    return None


def pick(reply: str, products: list) -> list:
    """Products the reply links to, in the order the reply mentions them."""
    text = reply or ""
    found = []
    for p in products:
        sku = str(p.get("sku") or "")
        if not sku:
            continue
        m = re.search(r"ozon\.ru/[^\s)\]>\"']*?(?<!\d)" + re.escape(sku) + r"(?!\d)", text)
        if m and all(f[1]["sku"] != sku for f in found):
            found.append((m.start(), p))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found][:MAX_CARDS]


def caption(i: int, p: dict, lang: str = "ru") -> str:
    ru = lang != "en"
    lines = [f"<b>{i}. {html.escape((p.get('name') or '?')[:200])}</b>"]
    price = _rub(p.get("price"))
    if p.get("old_price"):
        price += f"  <s>{_rub(p['old_price'])}</s>"
    if price:
        lines.append(("💰 " if ru else "💰 ") + price)
    if p.get("rating"):
        r = f"⭐ {p['rating']}"
        if p.get("reviews"):
            r += f" ({p['reviews']} " + (_reviews_ru(p["reviews"]) if ru else "reviews") + ")"
        lines.append(r)
    if p.get("delivery"):
        lines.append(("🚚 Доставка: " if ru else "🚚 Delivery: ") + html.escape(str(p["delivery"])))
    return "\n".join(lines)


def keyboard(p: dict, lang: str = "ru") -> dict:
    return {"inline_keyboard": [[{"text": "🛒 Открыть на Озоне" if lang != "en" else "🛒 Open on Ozon",
                                  "url": p["url"]}]]}


def _upload(bot, chat_id, img: bytes, cap: str, kb_json: str) -> bool:
    import requests
    r = requests.post(f"{bot._api}/sendPhoto",
                      data={"chat_id": chat_id, "caption": cap, "parse_mode": "HTML",
                            "reply_markup": kb_json},
                      files={"photo": ("card.jpg", img)}, timeout=60)
    if r.status_code != 200:
        logger.warning("sendPhoto HTTP %s: %s", r.status_code, r.text[:200])
    return r.status_code == 200 and bool(r.json().get("ok"))


def send_cards(bot, chat_id, reply: str, products: list, lang: str = "ru") -> int:
    """Send one card per linked product. Returns how many went out."""
    import json
    sent = 0
    for i, p in enumerate(pick(reply, products), 1):
        cap, kb = caption(i, p, lang), keyboard(p, lang)
        ok = False
        img = _photo_bytes(p["image"]) if p.get("image") else None
        if img:
            try:
                ok = _upload(bot, chat_id, img, cap, json.dumps(kb))
                if not ok:
                    logger.warning("card photo for %s refused by Telegram", p.get("sku"))
            except Exception as exc:
                logger.warning("card photo for %s failed: %s", p.get("sku"), exc)
        elif p.get("image"):
            logger.warning("card photo for %s: could not download %s", p.get("sku"), p["image"])
        if not ok:                         # no photo, or Telegram would not fetch it
            try:
                bot._send_text(chat_id, cap, parse_mode="HTML", keyboard=kb)
                ok = True
            except Exception as exc:
                logger.warning("card text for %s failed: %s", p.get("sku"), exc)
        sent += ok
    return sent
