"""Ozon (ozon.ru) for the buyer: search, product card, reviews.

Ported from eduard256/ozon-mcp-server (MIT, JavaScript) after comparing the
public Ozon MCP servers (2026-09-23). Ozon has no public API for buyers; the
seller-API servers answer a different question, and the extension-based one
needs the user's live Chrome profile. This approach needs neither: one real
browser passes the anti-bot challenge once, then every request is a fetch()
from inside that page to the site's own JSON endpoint

    https://www.ozon.ru/api/composer-api.bx/page/json/v2?url=<site path>

which returns the page as structured `widgetStates` -- no HTML scraping.

Differences from the original, all measured on this machine:
  * the browser is the system Edge (channel "msedge"), not the bundled
    headless shell -- that one is fingerprinted and got a 403 at once;
  * Playwright's sync API is thread-affine, and the bot calls from many
    threads, so the browser lives on ONE worker thread fed by a queue;
  * a blocked network is reported as such. From a datacenter/VPN address Ozon
    serves "Похоже, нет соединения" for every browser -- the fix is the
    network (OZON_PROXY, or route ozon.ru outside the VPN), not retries.

Parsers are pure functions over the composer JSON so they are unit-tested on
saved samples (tests/samples/ozon_*.json) without a browser.
"""
from __future__ import annotations

import json
import turn_trace
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import queue
import re
import threading
import time
import urllib.parse

logger = logging.getLogger("assistant.ozon")

HOME = "https://www.ozon.ru/"
API = "https://www.ozon.ru/api/composer-api.bx/page/json/v2?url="
CHALLENGE_WAIT_S = _cfg_env.env_float("OZON_CHALLENGE_WAIT", 12)
IDLE_CLOSE_S = 10 * 60
CALL_TIMEOUT_S = 150
# "http://user:pass@host:port" -- a Russian residential/mobile exit. Ozon
# refuses datacenter and VPN addresses outright.
OZON_PROXY = os.getenv("OZON_PROXY", "").strip()
# Our own Chromium copy under a unique exe name, so the VPN's split tunnel can
# exclude exactly this browser (and nothing else) -- ozon.ru blocks the VPN exit.
OZON_EXE = os.getenv("OZON_BROWSER_EXE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "ozon_chromium", "ozon-chromium.exe"))
OZON_CHANNEL = os.getenv("OZON_BROWSER_CHANNEL", "msedge").strip()
OZON_HEADLESS = os.getenv("OZON_HEADLESS", "0").strip() not in ("0", "false", "no")
# Parallel browsers for one shopping run (user, 2026-09-28: "8 воркеров"). Each is
# its own Chromium on its own thread; 16 in a row were not blocked.
OZON_WORKERS = max(1, _cfg_env.env_int("OZON_WORKERS", 8))

SORTS = {"popular": "", "price": "price", "price_desc": "price_desc",
         "rating": "rating", "new": "new", "discount": "discount"}

_BLOCK_RE = re.compile(r"нет соединения|доступ ограничен|antibot|ограничен", re.I)


class OzonError(Exception):
    """Written to be shown to the model and, through it, to the user."""


class OzonBlocked(OzonError):
    pass


class OzonAgeGate(OzonError):
    """18+ goods: Ozon shows them only after a date of birth is entered. The
    search page then lists nothing of the kind, and other wordings of the same
    need drift to look-alikes (a bath plug for an anal plug)."""


def _age_gated(page: dict) -> bool:
    return any(_widget_name(k) == "userAdultModal" for k in (page or {}).get("widgetStates", {}))


# ── pure parsers (port of parse.js) ────────────────────────────────────────────

def _widget_name(key: str) -> str:
    return str(key).split("-")[0]


def _widget(page: dict, name: str):
    for k, v in (page or {}).get("widgetStates", {}).items():
        if _widget_name(k) == name:
            try:
                return json.loads(v)
            except (TypeError, ValueError):
                return None
    return None


def _widgets(page: dict, name: str) -> list:
    out = []
    for k, v in (page or {}).get("widgetStates", {}).items():
        if _widget_name(k) == name:
            try:
                out.append(json.loads(v))
            except (TypeError, ValueError):
                pass
    return out


def price_to_number(text):
    """"53 022 ₽" -> 53022; garbage -> None."""
    if not isinstance(text, str):
        return None
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def clean_url(link):
    if not link:
        return None
    path = str(link).split("?")[0]
    return path if path.startswith("http") else "https://www.ozon.ru" + path


def sku_from_url(url):
    s = str(url or "")
    m = re.search(r"-(\d+)/?(?:\?|$)", s) or re.search(r"(\d{6,})", s)
    return m.group(1) if m else None


_BADGE = re.compile(r"^(стало дешевле|оригинал|хит|новинка|акция|распродажа|выбор|"
                    r"бестселлер|ozon|premium|самовывоз|скидка)", re.I)


_MONTHS = {m: i for i, m in enumerate(("января", "февраля", "марта", "апреля", "мая", "июня",
                                         "июля", "августа", "сентября", "октября", "ноября",
                                         "декабря"), 1)}


def delivery_days(text, today=None):
    """'12 июня' / 'завтра' / 'послезавтра' / 'сегодня' -> days from today, else None."""
    import datetime as _dt
    t = str(text or "").strip().lower()
    today = today or _dt.date.today()
    for word, n in (("сегодня", 0), ("послезавтра", 2), ("завтра", 1)):
        if t.startswith(word):
            return n
    m = re.match(r"(\d{1,2})\s+([а-я]+)", t)
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        d = _dt.date(today.year, _MONTHS[m.group(2)], int(m.group(1)))
    except ValueError:
        return None
    if d < today:                     # "5 января" seen in December
        d = d.replace(year=today.year + 1)
    return (d - today).days


def _search_item(it: dict):
    if not it:
        return None
    ms = it.get("mainState") or []
    price_block = next((s.get("priceV2") for s in ms if s.get("type") == "priceV2"), None) or {}
    prices = price_block.get("price") or []
    price = price_to_number(next((p.get("text") for p in prices if p.get("textStyle") == "PRICE"), None))
    old = price_to_number(next((p.get("text") for p in prices
                                if p.get("textStyle") == "ORIGINAL_PRICE"), None))
    name = next(((s.get("textDS") or {}).get("text") for s in ms if s.get("id") == "name"), None)
    rating = reviews = None
    rating_list = next(((s.get("labelListV2") or {}).get("items") for s in ms
                        if s.get("labelListV2") and "ic_s_star" in json.dumps(s["labelListV2"])), None)
    if isinstance(rating_list, list):
        texts = [(x.get("text") or {}).get("text") for x in rating_list if x.get("type") == "text"]
        if texts and texts[0]:
            try:
                rating = float(str(texts[0]).replace(",", "."))
            except ValueError:
                pass
        if len(texts) > 1 and texts[1]:
            reviews = price_to_number(texts[1])
    brand = None
    for s in ms:
        ll = s.get("labelListV2")
        if not ll or "ic_s_star" in json.dumps(ll):
            continue
        cand = next(((x.get("text") or {}).get("text") for x in ll.get("items") or []
                     if x.get("type") == "text"), None)
        if cand and cand.strip() and not _BADGE.match(cand.strip()):
            brand = cand.strip()
            break
    url = clean_url((it.get("action") or {}).get("link"))
    sku = str(it.get("sku") or it.get("id") or sku_from_url(url) or "") or None
    img = None
    for x in (it.get("tileImage") or {}).get("items") or []:
        if (x.get("image") or {}).get("link"):
            img = x["image"]["link"]
            break
    if not sku or not price:
        return None
    atc = (((it.get("multiButton") or {}).get("ozonButton") or {}).get("addToCart") or {})
    delivery = (atc.get("title") or (atc.get("actionButton") or {}).get("title")) or None
    if delivery and not re.search(r"\d|завтра|сегодня|послезавтра|today|tomorrow", delivery, re.I):
        delivery = None                 # the bare button caption «В корзину», not a date
    return {"sku": sku, "name": name, "price": price, "delivery": delivery,
            "old_price": old if old and old > price else None,
            "discount": price_block.get("discount"), "rating": rating,
            "reviews": reviews, "brand": brand, "url": url,
            "image": img or (it.get("tileImage") or {}).get("coverImage")}


def parse_search(page: dict, limit: int = 12) -> list:
    grid = _widget(page, "tileGridDesktop") or {}
    items = [x for x in (_search_item(i) for i in grid.get("items") or []) if x]
    return items[:limit]


def _rs_text(arr) -> str:
    return re.sub(r"\s+", " ", " ".join(
        str(v.get("text") or v.get("content")) for v in (arr or [])
        if isinstance(v, dict) and (v.get("text") or v.get("content")))).strip()


def _score(page: dict) -> tuple:
    w = _widget(page, "webSingleProductScore") or _widget(page, "webReviewProductScore")
    text = (w or {}).get("text") if isinstance(w, dict) and isinstance(w.get("text"), str) \
        else json.dumps(w or {}, ensure_ascii=False)
    rating = reviews = None
    ws = _widget(page, "webReviewProductScore")
    if isinstance(ws, dict) and isinstance(ws.get("totalScore"), (int, float)):
        # Live pages carry the numbers themselves; a whole score (5) has no
        # "d.d" to find in the text.
        return float(ws["totalScore"]), ws.get("reviewsCount")
    m = re.search(r"(\d[.,]\d)", text)
    if m:
        rating = float(m.group(1).replace(",", "."))
    m = re.search(r"(\d[\d\s]*)\s*отзыв", text)
    if m:
        reviews = price_to_number(m.group(1))
    return rating, reviews


def parse_description(page2: dict) -> str:
    # A second webDescription carries text blocks ("Комплектация: ...") that the
    # original ignores; on image-only descriptions it is the only text there is.
    extra = []
    for x in _widgets(page2, "webDescription"):
        for c in x.get("characteristics") or []:
            if isinstance(c, dict) and isinstance(c.get("content"), str):
                extra.append(f"{c.get('title') or ''}: {c['content']}".strip(": "))
    w = next((x for x in _widgets(page2, "webDescription") if x.get("richAnnotationJson")), None)
    if not w:
        return re.sub(r"\s+", " ", " ".join(extra)).strip()
    ra = w["richAnnotationJson"]
    if isinstance(ra, str):
        try:
            ra = json.loads(ra)
        except ValueError:
            return ""
    texts = []

    def walk(n):
        if isinstance(n, list):
            for x in n:
                walk(x)
        elif isinstance(n, dict):
            if n.get("type") == "text" and isinstance(n.get("content"), str):
                texts.append(n["content"])
            for v in n.values():
                if isinstance(v, (dict, list)):
                    walk(v)
    walk(ra.get("content", ra) if isinstance(ra, dict) else ra)
    return re.sub(r"\s+", " ", " ".join(dict.fromkeys(texts + extra))).strip()


def parse_details(base: dict, page2: dict | None = None) -> dict:
    heading = _widget(base, "webProductHeading") or {}
    price = _widget(base, "webPrice") or {}
    gallery = _widget(base, "webGallery") or {}
    seo_link = (((base or {}).get("seo") or {}).get("link") or [{}])[0].get("href")
    sku = str(gallery.get("sku") or "") or sku_from_url(seo_link)
    rating, reviews = _score(base)
    chars = {}
    for c in (_widget(base, "webShortCharacteristics") or {}).get("characteristics") or []:
        title = _rs_text((c.get("title") or {}).get("textRs")) if isinstance(c.get("title"), dict) \
            else (c.get("title") if isinstance(c.get("title"), str) else "")
        value = _rs_text(c.get("values") or c.get("contentRS") or c.get("valueRs"))
        if title and value:
            chars[title] = value
    seller = _widget(base, "webCurrentSeller") or {}
    seller_name = (((seller.get("sellerCell") or {}).get("centerBlock") or {}).get("title") or {}).get("text") \
        or (seller.get("title") or {}).get("text") if seller else None
    images = []
    if gallery.get("coverImage"):
        images.append(gallery["coverImage"])
    for im in gallery.get("images") or []:
        src = im.get("src") or im.get("image") if isinstance(im, dict) else im
        if isinstance(src, str):
            images.append(src)
    return {
        "sku": sku or None,
        "name": heading.get("title") or ((base or {}).get("seo") or {}).get("title"),
        "url": clean_url(seo_link) or (f"https://www.ozon.ru/product/{sku}/" if sku else None),
        "price": price_to_number(price.get("cardPrice")) or price_to_number(price.get("price")),
        "price_regular": price_to_number(price.get("price")),
        "old_price": price_to_number(price.get("originalPrice")),
        "available": price.get("isAvailable"),
        "rating": rating, "reviews": reviews,
        "seller": seller_name or None,
        "images": list(dict.fromkeys(images))[:10],
        "characteristics": chars,
        "description": parse_description(page2 or {})[:3000],
    }


def review_photo_urls(content: dict) -> list:
    """Buyer photos of one review; the item shape varies (str, {url|src|link|image})."""
    out = []
    for ph in (content or {}).get("photos") or []:
        u = ph if isinstance(ph, str) else next(
            (ph.get(k) for k in ("url", "src", "link", "image", "original", "preview")
             if isinstance(ph, dict) and isinstance(ph.get(k), str)), None)
        if u and u.startswith("http"):
            out.append(u)
    return out


def parse_reviews(page: dict, limit: int = 10, sku=None) -> dict:
    """`sku`: keep only reviews of THAT item. Ozon lists the whole family's
    reviews on every variant's page (live 2026-09-27: a snow blower was judged by
    its siblings' reviews) -- each review carries its own itemId."""
    w = _widget(page, "webListReviews") or {}
    raw = w.get("reviews") or w.get("items") or []
    if sku:
        raw = [r for r in raw if str(r.get("itemId") or sku) == str(sku)]
    rating, total = _score(page)
    out = []
    for r in raw[:limit]:
        c = r.get("content") or {}
        a = r.get("author") or {}
        ts = r.get("publishedAt") or r.get("createdAt")
        out.append({
            "author": a.get("title") or " ".join(x for x in (a.get("firstName"), a.get("lastName")) if x)
            or ("Аноним" if r.get("isAnonymous") else None),
            "score": c.get("score") if isinstance(c.get("score"), (int, float)) else None,
            "comment": c.get("comment") or "", "pros": c.get("positive") or "",
            "cons": c.get("negative") or "",
            "date": time.strftime("%Y-%m-%d", time.gmtime(ts)) if isinstance(ts, (int, float)) else None,
            "purchased": r.get("isItemPurchased"),
            "photos": review_photo_urls(c),
        })
    nxt = [l.get("urlParams") for l in (w.get("paging") or {}).get("links") or []
           if str(l.get("text")) == str(int((w.get("paging") or {}).get("page") or 1) + 1)]
    return {"rating": rating, "total_reviews": total, "reviews": out,
            "next": nxt[0] if nxt else None}


_GENERIC_FILTERS = {"category", "delivery", "currency_price", "is_promo", "seller", "isdiscount",
                    "has_points_from_reviews", "brandcertified", "color", "rating", "premium"}


_PROMO_FILTER = re.compile(r"(?i)рассрочк|морков|бал+ы|кешб|скидк|распрод|акци|premium|ozon|express|fresh|оригинал|рейтинг|официальн")


def parse_filters(page: dict, max_values: int = 8) -> list:
    """The category's own filters from a search page: [{key, title, values}].
    What a shop assistant would ask about (самоходный? стартер? ширина ковша?)
    -- generic ones (price, delivery, seller, promos) are left out."""
    out = []
    for w in _widgets(page, "filtersDesktop"):
        for sec in (w or {}).get("sections") or []:
            for f in sec.get("filters") or []:
                t, key = f.get("type"), f.get("key")
                body = f.get(t) or {}
                if (key in _GENERIC_FILTERS or not body.get("title")
                        or _PROMO_FILTER.search(str(body["title"]))):
                    continue
                vals = []
                for sub in body.get("sections") or []:
                    for it in sub.get("items") or []:
                        tt = it.get("title")
                        tt = tt.get("text") if isinstance(tt, dict) else tt
                        if tt:
                            vals.append(str(tt))
                if t == "boolFilter":
                    vals = ["да"]
                if vals:
                    out.append({"key": key, "title": body["title"], "values": vals[:max_values]})
    return out


def search_filters(query: str) -> list:
    return parse_filters(fetch_page(search_path(query)))


def resolve_short(url: str) -> str:
    """ozon.ru/t/<code> (the app's share link) -> /product/.../ . A plain HTTP
    request gets 501 from the anti-bot wall; the browser follows the redirect.
    Live 10-01: the third of three shared links was never opened and the agent
    found a product by guessing its name instead."""
    def go(page):
        page.goto(url, wait_until="domcontentloaded", timeout=90000)
        return page.url
    path = urllib.parse.urlparse(_BROWSER.fetch(go)).path
    if not path.startswith("/product/"):
        raise OzonError(f"the share link {url} does not lead to a product")
    return path if path.endswith("/") else path + "/"


def product_path(product: str) -> str:
    """sku ("1185261285"), full url, or slug -> "/product/.../"."""
    p = str(product or "").strip()
    if not p:
        raise OzonError("product is required: a link, an SKU number or a product slug")
    if re.match(r"^(?:www\.)?ozon\.ru/", p):
        p = "https://" + p
    if re.match(r"^https?://", p):
        path = urllib.parse.urlparse(p).path
        if path.startswith("/t/"):         # a share link: only the site knows the card
            path = resolve_short(p)
        return path if path.endswith("/") else path + "/"
    if p.startswith("/product/"):
        return p if p.endswith("/") else p + "/"
    if p.isdigit():
        return f"/product/{p}/"
    return "/product/" + p.strip("/") + "/"


def search_path(query: str, sort: str = "popular", price_min=None, price_max=None, page: int = 1) -> str:
    if not str(query or "").strip():
        raise OzonError("query is required")
    url = "/search/?text=" + urllib.parse.quote(query) + "&from_global=true"
    if SORTS.get(sort):
        url += "&sorting=" + SORTS[sort]
    if price_min is not None or price_max is not None:
        url += "&currency_price=%s.000%%3B%s.000" % (int(price_min or 0), int(price_max or 99999999))
    if page and int(page) > 1:
        url += "&page=%d" % int(page)
    return url


# ── the browser, on one thread ──────────────────────────────────────────────────
#
# One Chromium process, one CONTEXT per user. A context is a separate cookie jar,
# and Ozon keeps the chosen pickup point in the anonymous session's cookies -- so
# with one shared jar, user A choosing Казань moved user B's delivery dates too.
# Each user's jar is saved to a file (their sandbox's .agent/ozon_state.json) so
# the choice and the passed anti-bot check survive a restart of the bot.

MAX_CONTEXTS = 4                       # least-recently used ones are saved and closed

_OWNER = threading.local()


class use_owner:
    """`with use_owner(key, state_file):` -- Ozon calls in this block run in that
    user's own browser context."""

    def __init__(self, key: str, state_file: str = ""):
        self.key, self.state_file = str(key or ""), state_file

    def __enter__(self):
        self._prev = getattr(_OWNER, "v", None)
        _OWNER.v = (self.key, self.state_file)
        return self

    def __exit__(self, *exc):
        _OWNER.v = self._prev
        return False


def _default_state_file() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "ozon_state.json")


def current_owner() -> tuple:
    return getattr(_OWNER, "v", None) or ("", _default_state_file())


class _Browser:
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._thread = None
        self._lock = threading.Lock()
        self.gen = 0                       # bumped by reset(): drop contexts, re-read the jar

    def reset(self):
        self.gen += 1

    def fetch(self, path) -> dict:
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="ozon-browser", daemon=True)
                self._thread.start()
        box: dict = {}
        done = threading.Event()
        self._q.put((current_owner(), path, box, done))
        if not done.wait(CALL_TIMEOUT_S):
            raise OzonError("Ozon did not answer in time")
        if "error" in box:
            raise box["error"]
        return box["data"]

    def _run(self):
        pw = browser = None
        ctxs: dict = {}                    # key -> [context, page, challenged, state_file]
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._drain(OzonError("playwright is not installed (pip install playwright)"))
            return
        pw = sync_playwright().start()
        seen_gen = self.gen
        try:
            while True:
                try:
                    (key, state_file), path, box, done = self._q.get(timeout=IDLE_CLOSE_S)
                except queue.Empty:
                    break                          # idle: free the RAM, relaunch on demand
                if seen_gen != self.gen:           # another browser changed the jar (pickup point)
                    for slot in ctxs.values():     # closed WITHOUT saving: ours is stale
                        try:
                            slot[0].close()
                        except Exception:
                            pass
                    ctxs.clear()
                    seen_gen = self.gen
                for attempt in range(2):
                    try:
                        if browser is None or not browser.is_connected():
                            browser = self._launch(pw)
                            ctxs.clear()
                        slot = ctxs.pop(key, None)
                        if slot is None:
                            slot = self._context(browser, state_file)
                        ctxs[key] = slot           # most recently used last
                        while len(ctxs) > MAX_CONTEXTS:
                            self._close(ctxs.pop(next(iter(ctxs))))
                        page = slot[1]
                        if not slot[2]:
                            self._challenge(page)
                            slot[2] = True
                            _hide_from_taskbar()       # the window exists only after the first goto
                        if callable(path):         # a UI action on the live page
                            box["data"] = path(page)
                            self._save(slot)
                            break
                        status, text = page.evaluate(
                            "async u => { const r = await fetch(u, {headers: {accept: 'application/json'}});"
                            " return [r.status, await r.text()]; }",
                            API + urllib.parse.quote(path, safe=""))
                        if status == 200:
                            box["data"] = json.loads(text)
                            break
                        if status in (403, 307) and attempt == 0:
                            self._close(ctxs.pop(key))
                            continue
                        raise OzonError(f"Ozon returned HTTP {status}")
                    except OzonError as exc:
                        box["error"] = exc
                        break
                    except Exception as exc:       # dead browser, navigation timeout
                        self._close(ctxs.pop(key, None))
                        if "net::ERR_" in str(exc):
                            # The network itself cannot reach ozon.ru (the VPN exit
                            # drops it) -- a second 90 s attempt only doubles the wait.
                            box["error"] = OzonBlocked(
                                "ozon.ru is unreachable from this server's network (%s). "
                                "Ozon blocks VPN and datacenter addresses; it needs "
                                "OZON_PROXY (a Russian home/mobile proxy) or ozon.ru "
                                "routed outside the VPN." % str(exc).split(" at ")[0][:80])
                            break
                        if attempt:
                            box["error"] = OzonError(f"browser failed: {exc}")
                done.set()
        finally:
            for slot in ctxs.values():
                self._close(slot)
            try:
                if browser is not None:
                    browser.close()
            except Exception:
                pass
            try:
                pw.stop()
            except Exception:
                pass

    def _launch(self, pw):
        kw = dict(channel=OZON_CHANNEL or None, headless=OZON_HEADLESS,
                  ignore_default_args=["--enable-automation"],
                  args=["--disable-blink-features=AutomationControlled",
                        "--window-position=-32000,-32000", "--mute-audio"])
        if OZON_EXE and os.path.isfile(OZON_EXE):
            kw.pop("channel")
            kw["executable_path"] = OZON_EXE
        if OZON_PROXY:
            kw["proxy"] = {"server": OZON_PROXY}
        return pw.chromium.launch(**kw)

    @staticmethod
    def _context(browser, state_file: str) -> list:
        kw = dict(locale="ru-RU", viewport={"width": 1366, "height": 900})
        if state_file and os.path.isfile(state_file):
            kw["storage_state"] = state_file
        ctx = browser.new_context(**kw)
        page = ctx.new_page()
        _hide_from_taskbar()
        return [ctx, page, False, state_file]

    @staticmethod
    def _save(slot) -> None:
        try:
            if slot and slot[3]:
                os.makedirs(os.path.dirname(slot[3]), exist_ok=True)
                slot[0].storage_state(path=slot[3])
        except Exception as exc:
            logger.warning("could not save the Ozon session to %s: %s", slot[3], exc)

    @staticmethod
    def _challenge(page):
        page.goto(HOME, wait_until="domcontentloaded", timeout=90000)
        deadline = time.time() + CHALLENGE_WAIT_S * 3
        page.wait_for_timeout(int(CHALLENGE_WAIT_S * 1000))
        while _BLOCK_RE.search(page.title() or "") and time.time() < deadline:
            page.wait_for_timeout(3000)
        title = page.title() or ""
        if _BLOCK_RE.search(title):
            raise OzonBlocked(
                "Ozon refuses this network (its page says: '%s'). It blocks VPN and "
                "datacenter addresses. Set OZON_PROXY to a Russian home/mobile proxy "
                "or route ozon.ru outside the VPN." % title[:60])

    @classmethod
    def _close(cls, slot):
        if not slot:
            return
        cls._save(slot)
        try:
            slot[0].close()
        except Exception:
            pass

    def _drain(self, err):
        while True:
            try:
                _o, _p, box, done = self._q.get_nowait()
            except queue.Empty:
                return
            box["error"] = err
            done.set()


def _hide_from_taskbar() -> None:
    """The browsers sit at -32000,-32000 but still had a taskbar button each
    (user: "эту прелисть хайдить"). Headless is not an option -- Ozon blocks it
    (measured 2026-09-28, both old and --headless=new). Tool-window style drops
    the button; only our off-screen windows are touched."""
    if os.name != "nt" or OZON_HEADLESS:
        return
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        GWL_EXSTYLE, TOOL, APP = -20, 0x80, 0x40000
        found = []

        def cb(hwnd, _):
            r = wintypes.RECT()
            u.GetWindowRect(hwnd, ctypes.byref(r))
            if r.left <= -10000 and u.IsWindowVisible(hwnd):   # Chrome clamps -32000 to ~-10923
                cls = ctypes.create_unicode_buffer(64)
                u.GetClassNameW(hwnd, cls, 64)
                if cls.value.startswith("Chrome_WidgetWin"):
                    found.append(hwnd)
            return True
        u.EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(cb), 0)
        for h in found:
            st = u.GetWindowLongW(h, GWL_EXSTYLE)
            if st & TOOL and not st & APP:
                continue
            u.ShowWindow(h, 0)                                 # SW_HIDE, restyle, SW_SHOWNOACTIVATE
            u.SetWindowLongW(h, GWL_EXSTYLE, (st | TOOL) & ~APP)
            u.ShowWindow(h, 4)
    except Exception as exc:
        logger.debug("taskbar hide failed: %s", exc)


_BROWSERS = [_Browser() for _ in range(OZON_WORKERS)]
_BROWSER = _BROWSERS[0]                  # UI actions (the pickup point) run here
_NEXT = [0]
_NEXT_LOCK = threading.Lock()


def fetch_page(path: str) -> dict:
    with _NEXT_LOCK:
        b = _BROWSERS[_NEXT[0] % len(_BROWSERS)]
        _NEXT[0] += 1
    return b.fetch(path)


def pmap(fn, items) -> list:
    """fn over items on OZON_WORKERS threads, in order, each in the caller's
    user context (the owner is thread-local and would otherwise be lost)."""
    items = list(items)
    if len(items) < 2 or OZON_WORKERS < 2:
        return [fn(x) for x in items]
    owner = getattr(_OWNER, "v", None)

    def run(x):
        _OWNER.v = owner
        return fn(x)
    with turn_trace.Pool(min(OZON_WORKERS, len(items))) as ex:
        return list(ex.map(run, items))


_ACCESSORY_HEAD = re.compile(r"^\s*(комплектующ|запчаст|аксессуар|насадк|прокладк|уплотнит|"
                             r"чехол|шланг|кран|термометр|фильтр|переходник|крышк)", re.I)


def _stems(query: str) -> list:
    return [w[:max(4, len(w) - 3)] for w in re.findall(r"[а-яёa-z0-9]+", query.lower()) if len(w) > 3]


def is_accessory(name: str, query: str) -> bool:
    """'Комплектующие для самогонного аппарата' / 'Термометр ... для самогонного
    аппарата' is not a самогонный аппарат -- unless the user asked for the part."""
    n = str(name or "").lower()
    q = query.lower()
    if _ACCESSORY_HEAD.match(n) and not _ACCESSORY_HEAD.match(q):
        return True
    stems = _stems(query)
    m = re.search(r"\bдля\b(.*)", n)
    head = n[:m.start()] if m else n
    return bool(m and stems and any(s in m.group(1) for s in stems)
                and not any(s in head for s in stems))


LAST_LOCATION = {"text": ""}     # the header address seen on the last search


def _birthdate() -> str:
    """ДД.ММ.ГГГГ the user allowed for Ozon's 18+ gate (OZON_BIRTHDATE or
    runtime/ozon_birthdate.txt, never in the repo); "" = do not confirm."""
    v = os.getenv("OZON_BIRTHDATE", "").strip()
    if not v:
        f = os.path.join(os.path.dirname(_default_state_file()), "ozon_birthdate.txt")
        if os.path.isfile(f):
            v = open(f, encoding="utf-8").read().strip()
    return v if re.fullmatch(r"\d\d\.\d\d\.\d{4}", v) else ""


def confirm_age(path: str) -> None:
    """Fill Ozon's «Подтвердите возраст» modal in the session's browser; the
    cookie it sets is saved with the session, every browser re-reads it."""
    born = _birthdate()

    def fill(page):
        page.goto(HOME.rstrip("/") + path, wait_until="domcontentloaded", timeout=90000)
        box = page.locator("input[inputmode=numeric]").first
        box.wait_for(timeout=20000)
        box.click(force=True)
        want = born.replace(".", "")
        # the mask swallows the key typed right after an auto-inserted dot:
        # type digit by digit and retype whatever did not land
        for _ in range(len(want) * 2):
            got = re.sub(r"\D", "", box.input_value())
            if got == want or not want.startswith(got):
                break
            page.keyboard.type(want[len(got)])
            page.wait_for_timeout(150)
        if re.sub(r"\D", "", box.input_value()) != want:
            raise OzonError("Ozon's age form did not take the birth date")
        page.locator("button", has_text="Подтвердить").first.click(timeout=20000, force=True)
        page.wait_for_timeout(4000)
        return True
    _BROWSER.fetch(fill)
    for b in _BROWSERS[1:]:
        b.reset()


def _search_page(path: str, query: str) -> dict:
    page = fetch_page(path)
    if _age_gated(page) and _birthdate():
        with _AGE_LOCK:                    # parallel wordings: confirm once
            page = fetch_page(path)
            if _age_gated(page):
                confirm_age(path)
                page = fetch_page(path)
    if _age_gated(page):
        raise OzonAgeGate(f"'{query}' is an 18+ category: Ozon hides it until the age is confirmed")
    return page


_AGE_LOCK = threading.Lock()


def image_search_path(image_path: str) -> str:
    """Upload a picture to Ozon's photo search (the camera in the search bar)
    and return its results path, /search-by-image?image_id=... . Ozon asks to
    frame the product first; the whole picture is kept."""
    if not os.path.isfile(image_path or ""):
        raise OzonError("no picture to search by")

    def upload(page):
        page.goto(HOME, wait_until="domcontentloaded", timeout=90000)
        camera = page.locator("[data-widget=searchBarDesktop] input[name=text] ~ div button").first
        for _ in range(10):                # a click before the page wakes up does nothing
            camera.click(timeout=20000)
            page.wait_for_timeout(1500)
            if page.locator("input[type=file]").count():
                break
        page.locator("input[type=file]").first.set_input_files(image_path)
        page.locator("button", has_text="Найти").last.click(timeout=20000)
        page.wait_for_url(re.compile(r"/search-by-image"), timeout=30000)
        return page.url.split("ozon.ru", 1)[1]
    return _BROWSER.fetch(upload)


def search_by_image(results_path: str, limit: int = 10, page: int = 1) -> list:
    """Photo results come in likeness order only: a sorting= turns them into the
    whole catalogue by price/rating (measured 10-01). The priority is the
    shopper's to apply."""
    url = results_path
    if page and int(page) > 1:
        url += "&page=%d" % int(page)
    data = _search_page(url, "photo")
    LAST_LOCATION["text"] = parse_location(data)
    return parse_search(data, 60)[:limit]


def search(query: str, sort: str = "popular", price_min=None, price_max=None,
           limit: int = 10, max_delivery_days=None, page: int = 1) -> list:
    if sort == "price":
        # Ozon's own price sort floats every 98 ₽ spare part to the top. Take the
        # relevance ranking instead, drop parts/accessories, then order by price.
        page = _search_page(search_path(query, "popular", price_min, price_max, page), query)
        LAST_LOCATION["text"] = parse_location(page)
        items = parse_search(page, 60)
        items = sorted((x for x in items if not is_accessory(x.get("name"), query)),
                       key=lambda x: x["price"])
    else:
        page = _search_page(search_path(query, sort, price_min, price_max, page), query)
        LAST_LOCATION["text"] = parse_location(page)
        items = parse_search(page, 60)
    if max_delivery_days is not None:
        items = [x for x in items
                 if (delivery_days(x.get("delivery")) is not None
                     and delivery_days(x.get("delivery")) <= int(max_delivery_days))]
    return items[:limit]


def details(product: str) -> dict:
    path = product_path(product)
    base = fetch_page(path)
    try:
        page2 = fetch_page(path + "?layout_container=pdpPage2column&layout_page_index=2")
    except OzonError:
        page2 = {}
    return parse_details(base, page2)


def reviews(product: str, limit: int = 10, pages: int = 4) -> dict:
    """Reviews of this exact item only, walking up to `pages` pages to fill `limit`.
    rating/total_reviews stay the family's (that is all Ozon shows);
    own_only=True tells the reader the listed ones are this variant's."""
    path = product_path(product) + "reviews/"
    sku = sku_from_url(path)
    r = parse_reviews(fetch_page(path), limit, sku)
    got, n = r["reviews"], 1
    while len(got) < limit and r.get("next") and n < pages:
        r2 = parse_reviews(fetch_page(path + r["next"]), limit - len(got), sku)
        got += r2["reviews"]; r["next"] = r2.get("next"); n += 1
    r["reviews"], r["own_only"] = got, bool(sku)
    return r


# ── delivery location (pickup point) ────────────────────────────────────────────
#
# Without an address Ozon picks the region from the exit IP. Choosing a pickup
# point is a site action, not a composer page: a GET of the modal's "Заберу
# отсюда" link changes nothing. What works (measured 2026-09-23): the point's
# public page /geo/<city>/<id>/ has "Сохранить адрес и перейти к покупкам";
# clicking it in the browser stores the point in the anonymous session, and
# every later search/card is priced and dated for it. No account involved.

def parse_location(page: dict) -> str:
    """The header's address bar: 'Санкт-Петербург' or 'Пункт Ozon • ул. ...'."""
    raw = (page or {}).get("widgetStates", {})
    for k, v in raw.items():
        if _widget_name(k) == "addressBookBarWeb":
            texts = re.findall(r'"text":"([^"]+)"', v or "")
            texts = [t for t in texts if t not in ("&bull;", "Укажите адрес")]
            return " ".join(texts[:2]).strip()
    return ""


def current_location() -> str:
    return parse_location(fetch_page("/"))


def geocode(place: str):
    """(lat, lon, label) for a city or street address, via OpenStreetMap."""
    import requests
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": place, "format": "json", "limit": 1, "countrycodes": "ru",
                             "accept-language": "ru"},
                     headers={"User-Agent": "f5-assistant-ozon/1.0"}, timeout=20)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise OzonError(f"could not find the place '{place}' on the map")
    return float(js[0]["lat"]), float(js[0]["lon"]), js[0].get("display_name", place)


def _map_objects(page: dict) -> list:
    m = _widget(page, "addressEditMap") or {}
    return (((m.get("markerBundle") or {}).get("mapObjectCollection") or {})
            .get("mapObjects") or [])


def nearest_point(lat: float, lon: float) -> str | None:
    """Pickup point id nearest to (lat, lon); clusters are opened until a pin shows."""
    path = f"/modal/commonDelivery?glr=t&lat={lat:.6f}&long={lon:.6f}&pid=4&pv=2&tab=pp"
    for _ in range(4):
        objs = _map_objects(fetch_page(path))
        if not objs:
            return None
        dist = lambda o: ((o["coordinates"]["latitude"] - lat) ** 2
                          + (o["coordinates"]["longitude"] - lon) ** 2)
        pins = [o for o in objs if o.get("type") == "PIN" and "pp=" in (o.get("actionLink") or "")]
        if pins:
            return re.search(r"pp=(\d+)", min(pins, key=dist)["actionLink"]).group(1)
        path = min(objs, key=dist).get("actionLink") or ""
        if not path:
            return None
    return None


def set_pickup_point(place: str) -> dict:
    """Make the pickup point nearest to `place` the delivery location."""
    lat, lon, label = geocode(place)
    pp = nearest_point(lat, lon)
    if not pp:
        raise OzonError(f"Ozon has no pickup point near '{place}'")

    def click(page):
        page.goto(f"{HOME}geo/x/{pp}/", wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(4000)
        btn = page.locator("button", has_text="Сохранить адрес").first
        btn.click(timeout=20000)
        page.wait_for_timeout(4000)
        return True

    _BROWSER.fetch(click)
    for b in _BROWSERS[1:]:
        b.reset()
    where = current_location()
    if "Пункт" not in where and "ул" not in where:
        raise OzonError(f"Ozon did not accept the pickup point (header still says '{where}')")
    return {"point_id": pp, "location": where, "searched": label}
