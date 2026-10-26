"""An Ozon price watch: the card is read on a schedule (reminders.py keeps the
schedule), and the user hears only of a real drop -- below the lowest price seen,
and down to their target if they named one. A failed read or a sold-out card says
nothing: «не подешевело» every few hours is spam.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("assistant.price_watch")


def _rub(n) -> str:
    return f"{int(n):,} ₽".replace(",", " ")


def read(item: dict) -> dict:
    """The watched card now, in the owner's Ozon session."""
    import ozon_client
    if item.get("key"):
        with ozon_client.use_owner(item["key"], item.get("state") or ""):
            return ozon_client.details(item["product"])
    return ozon_client.details(item["product"])


def check(item: dict, details=read) -> tuple:
    """(message or None, fields to store back on the watch)."""
    try:
        d = details(item)
    except Exception as exc:
        logger.warning("price watch %s: read failed: %s", item.get("product"), exc)
        return None, {}
    price, low, target = d.get("price"), item["price"], item.get("target")
    if not price or d.get("available") is False or price >= low or (target and price > target):
        return None, {}
    off = round(100 * (low - price) / low)
    return (f"📉 Подешевело: {item.get('name') or 'товар'}\n{_rub(low)} → {_rub(price)} (−{off}%)\n"
            f"{item['product']}"), {"price": price}
