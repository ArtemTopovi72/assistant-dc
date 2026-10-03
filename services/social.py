"""Social-wall reading for the research crawler.

Posters / afishas / event schedules for venues, theatres and cultural centres
usually live not on a static website but on the org's **VKontakte group** or
**Telegram channel**. Plain HTML scraping fails there:

- Telegram public channels ARE server-rendered at ``https://t.me/s/<name>`` and
  read auth-free.
- VK actively blocks scraping (serves a "BadBrowser" stub), so VK walls are read
  through the official VK API ``wall.get`` — which needs ``config.VK_TOKEN``.
  Without a token VK is skipped (the crawler falls back to Telegram / the site).

The crawler calls :func:`fetch_social_posts` for any social URL; it returns a
plain-text digest of the most recent posts (newest first) or ``None``.
"""
import logging
import re
from urllib.parse import urlparse

import requests

from config import (
    VK_TOKEN, VK_API_VERSION, SOCIAL_POSTS_LIMIT, DR_PAGE_TIMEOUT,
)

logger = logging.getLogger("assistant.social")

_MOBILE_UA = ("Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36")


# --------------------------------------------------------------------------- #
# URL classification
# --------------------------------------------------------------------------- #
def _host(url: str) -> str:
    try:
        h = urlparse(url).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def social_kind(url: str) -> str:
    """'vk', 'telegram', or '' for non-social URLs."""
    h = _host(url)
    if h in ("vk.com", "m.vk.com", "vk.ru"):
        return "vk"
    if h in ("t.me", "telegram.me", "telegram.dog"):
        return "telegram"
    return ""


def is_social_url(url: str) -> bool:
    return bool(social_kind(url))


# --------------------------------------------------------------------------- #
# Telegram — auth-free, server-rendered at t.me/s/<channel>
# --------------------------------------------------------------------------- #
def _tg_channel(url: str) -> str:
    """Extract the channel name from a t.me URL (drops /s/, post ids, query)."""
    path = urlparse(url).path.strip("/")
    parts = [p for p in path.split("/") if p]
    if parts and parts[0] == "s":
        parts = parts[1:]
    return parts[0] if parts else ""


def extract_telegram_posts(url: str, limit: int = SOCIAL_POSTS_LIMIT) -> str | None:
    channel = _tg_channel(url)
    if not channel or channel in ("joinchat", "+", "addstickers", "proxy"):
        return None
    try:
        from bs4 import BeautifulSoup
        r = requests.get(f"https://t.me/s/{channel}",
                         headers={"User-Agent": _MOBILE_UA, "Accept-Language": "ru,en"},
                         timeout=DR_PAGE_TIMEOUT)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        nodes = soup.select(".tgme_widget_message_text")
        if not nodes:
            return None
        # newest posts are last in the page; take the tail, present newest-first
        posts = []
        for n in nodes[-limit:]:
            txt = " ".join(n.get_text(" ").split())
            if txt:
                posts.append(txt)
        posts.reverse()
        if not posts:
            return None
        return (f"Telegram channel @{channel} — {len(posts)} most recent posts "
                f"(newest first):\n\n" + "\n\n— — —\n\n".join(posts))
    except Exception as exc:
        logger.debug("telegram fetch failed %s: %s", url, exc)
        return None


# --------------------------------------------------------------------------- #
# VK — via the official API (needs VK_TOKEN)
# --------------------------------------------------------------------------- #
def _vk_call(method: str, **params) -> dict | None:
    params["access_token"] = VK_TOKEN
    params["v"] = VK_API_VERSION
    try:
        r = requests.get(f"https://api.vk.com/method/{method}", params=params,
                         timeout=DR_PAGE_TIMEOUT)
        data = r.json()
        if "error" in data:
            logger.warning("VK API %s error: %s", method,
                           data["error"].get("error_msg"))
            return None
        return data.get("response")
    except Exception as exc:
        logger.debug("VK API %s failed: %s", method, exc)
        return None


def _vk_owner_id(url: str) -> int | None:
    """Resolve a VK group URL to a negative owner_id for wall.get."""
    path = urlparse(url).path.strip("/")
    screen = path.split("/")[0] if path else ""
    if not screen:
        return None
    m = re.fullmatch(r"(?:club|public|event)(\d+)", screen)
    if m:
        return -int(m.group(1))
    resolved = _vk_call("utils.resolveScreenName", screen_name=screen)
    if not resolved or "object_id" not in resolved:
        return None
    oid = int(resolved["object_id"])
    return -oid if resolved.get("type") in ("group", "page", "event") else oid


def extract_vk_posts(url: str, limit: int = SOCIAL_POSTS_LIMIT) -> str | None:
    if not VK_TOKEN:
        logger.info("VK wall %s skipped — no VK_TOKEN configured", url)
        return None
    owner = _vk_owner_id(url)
    if owner is None:
        return None
    resp = _vk_call("wall.get", owner_id=owner, count=limit, extended=0)
    if not resp or not resp.get("items"):
        return None
    posts = []
    for it in resp["items"][:limit]:
        txt = " ".join((it.get("text") or "").split())
        if not txt and it.get("copy_history"):  # reposts carry text in the original
            txt = " ".join((it["copy_history"][0].get("text") or "").split())
        if txt:
            posts.append(txt)
    if not posts:
        return None
    return (f"VK group {url} — {len(posts)} most recent wall posts (newest first):"
            f"\n\n" + "\n\n— — —\n\n".join(posts))


# --------------------------------------------------------------------------- #
# Unified entry
# --------------------------------------------------------------------------- #
def fetch_social_posts(url: str, limit: int = SOCIAL_POSTS_LIMIT) -> str | None:
    """Return a text digest of recent posts for a VK/Telegram URL, or None."""
    kind = social_kind(url)
    if kind == "telegram":
        return extract_telegram_posts(url, limit)
    if kind == "vk":
        return extract_vk_posts(url, limit)
    return None
