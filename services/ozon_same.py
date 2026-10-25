"""The same item from another seller: cheaper, sooner, or the best of both -- not a copy.

A copy on Ozon reuses the original's photos, so a picture match only finds
candidates. What tells them apart is the card's own characteristics: the eco
«same backpack» for 2 500 ₽ says «Полиуретан», 39x29 cm where the original
says «Кожа натуральная лакированная», 42x30 cm (2026-10-25). So:

    candidates  the card's «Есть дешевле или быстрее» list (sellers Ozon glued
                to this card) + Ozon's photo search on the card's first photo
    verify      every candidate's characteristics against the source's, in code:
                sizes within a centimetre, materials word for word, red-flag
                words (реплика, экокожа for a natural-leather source)
    rank        cheapest / soonest / the compromise on the price-days Pareto front

The model only words the answer; the verdicts are made here.
"""
from __future__ import annotations
from seams import dep

import logging
import os
import re
import tempfile

import ozon_client as _oz

logger = logging.getLogger("assistant.ozon.same")

OFFERS_CHECKED = 14        # cheapest and soonest of the glued sellers, each card opened
PHOTO_CHECKED = 10
CHEAP_RATIO = 0.5          # under half the source's price is the copy's price

SKIP_KEY = re.compile(r"(?i)страна|гарант|срок|артикул|код товара|упаковк|в упаковке|назначение")
_FAKE = re.compile(r"(?i)реплик|копи[яи]\b|\b1\s*:\s*1\b|под оригинал|люкс[- ]?копи|аналог")
_NOT_NATURAL = re.compile(r"(?i)эко|искусств|полиуретан|\bpu\b|кожзам|кож\.?\s*зам|синтет|имитац|под кожу")


def _words(v: str) -> set:
    return set(re.findall(r"[a-zа-я0-9]{2,}", str(v).lower().replace("ё", "е")))


_MEASURE = re.compile(r"\s*(-?\d+(?:[.,]\d+)?)\s*[a-zа-я\"]{0,6}\.?\s*", re.I)


def same_value(key: str, a: str, b: str) -> bool:
    ma, mb = _MEASURE.fullmatch(str(a)), _MEASURE.fullmatch(str(b))
    if ma and mb:
        na, nb = (float(m.group(1).replace(",", ".")) for m in (ma, mb))
        k = key.lower()
        tol = 1.0 if "см" in k else 10.0 if "мм" in k else max(abs(na), abs(nb)) * 0.03
        return abs(na - nb) <= tol + 1e-9
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return str(a).strip().lower() == str(b).strip().lower()
    return wa <= wb or wb <= wa or len(wa & wb) / len(wa | wb) >= 0.6


def compare(src: dict, cand: dict) -> dict:
    """{'verdict': same|different|unclear, 'matched': [keys], 'diff': [why], 'flags': [notes]}"""
    sc, cc = src.get("characteristics") or {}, cand.get("characteristics") or {}
    matched, diff, flags = [], [], []
    for k, v in sc.items():
        if SKIP_KEY.search(k) or k not in cc:
            continue
        (matched if same_value(k, v, cc[k]) else diff).append(k)
    diff = [f"{k}: «{cc[k]}» вместо «{sc[k]}»" for k in diff]
    text = " ".join([cand.get("name") or ""] + [str(x) for x in cc.values()])
    if _FAKE.search(text) and not _FAKE.search(src.get("name") or ""):
        diff.append("в карточке: «" + _FAKE.search(text).group() + "»")
    natural = any("натуральн" in str(v).lower() for v in sc.values())
    if natural and _NOT_NATURAL.search(text):
        diff.append("не натуральный материал: «" + _NOT_NATURAL.search(text).group() + "»")
    sp, cp = src.get("price"), cand.get("price")
    if sp and cp and cp < sp * CHEAP_RATIO:
        if not matched:                   # a 60 ₽ sticker the photo search found for a laptop
            diff.append(f"другой товар за {cp} ₽")
        flags.append(f"подозрительно дёшево: {cp} ₽ при {sp} ₽ у исходного")
    if cand.get("country") and cand["country"] != "RU":
        flags.append(f"продавец из-за рубежа ({cand['country']})")
    if not cand.get("reviews"):
        flags.append("нет отзывов")
    shared = sum(1 for k in sc if not SKIP_KEY.search(k))
    verdict = ("different" if diff else
               "same" if matched and len(matched) >= min(2, shared) else "unclear")
    return {"verdict": verdict, "matched": matched, "diff": diff, "flags": flags}


def _days(x):
    return _oz.delivery_days(x.get("delivery"))


def suspicious(x: dict) -> bool:
    return any(f.startswith("подозрительно") for f in (x.get("check") or {}).get("flags") or [])


def rank(pool: list) -> dict:
    """{'cheap': x, 'fast': x, 'balanced': x} over verified offers; one at a copy's price is
    listed, never picked."""
    pool = [x for x in pool if not suspicious(x)]
    if not pool:
        return {}
    big = 10 ** 6
    cheap = min(pool, key=lambda x: (x["price"], _days(x) if _days(x) is not None else big))
    dated = [x for x in pool if _days(x) is not None]
    if not dated:
        return {"cheap": cheap, "fast": None, "balanced": cheap}
    fast = min(dated, key=lambda x: (_days(x), x["price"]))
    front = [x for x in dated if not any(
        y["price"] <= x["price"] and _days(y) <= _days(x) and (y["price"] < x["price"] or _days(y) < _days(x))
        for y in dated)]
    p0, p1 = min(x["price"] for x in dated), max(x["price"] for x in dated)
    d0, d1 = min(_days(x) for x in dated), max(_days(x) for x in dated)
    score = lambda x: ((x["price"] - p0) / ((p1 - p0) or 1) + (_days(x) - d0) / ((d1 - d0) or 1), x["price"])
    return {"cheap": cheap, "fast": fast, "balanced": min(front, key=score)}


def _first_photo(ctx, url: str) -> str:
    from dr_urls import safe_get
    r = dep(ctx, "safe_get", safe_get)(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="ozon_same_")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path


def _card(ctx, sku: str) -> dict:
    return _oz.call(ctx, "parse_details", _oz.call(ctx, "fetch_page", _oz.call(ctx, "product_path", sku)))


def find_same(ctx, product: str, by_photo: bool = True, stage=lambda s: None) -> dict:
    """The source card, the verified same offers ranked, and the rejected ones with why."""
    stage("Ozon: opening the card")
    base = _oz.call(ctx, "fetch_page", _oz.call(ctx, "product_path", product))
    src = _oz.call(ctx, "parse_details", base)
    if not src.get("sku"):
        raise _oz.OzonError("could not open that product card")
    cands: dict = {}
    path = _oz.offers_path(base, "price")
    if path:
        stage("Ozon: other sellers of this item")
        offers = _oz.parse_offers(_oz.call(ctx, "fetch_page", path))
        dated = sorted((o for o in offers if _days(o) is not None), key=lambda o: (_days(o), o["price"]))
        for o in offers[:OFFERS_CHECKED] + dated[:OFFERS_CHECKED // 2]:
            cands.setdefault(o["sku"], dict(o, via="seller list"))
    if by_photo and src.get("images"):
        stage("Ozon: photo search")
        img = ""
        try:
            img = _first_photo(ctx, src["images"][0])
            found = _oz.call(ctx, "search_by_image", _oz.call(ctx, "image_search_path", img), 40)
            for x in found:
                if len([c for c in cands.values() if c["via"] == "photo"]) >= PHOTO_CHECKED:
                    break
                if x["sku"] == src["sku"]:     # the card itself: its date shows only on a tile
                    src["delivery"] = x.get("delivery")
                elif x["sku"] not in cands:
                    cands[x["sku"]] = dict(x, via="photo")
        except (_oz.OzonBlocked, _oz.OzonAgeGate):
            raise
        except Exception as exc:
            logger.warning("ozon_same photo search failed: %s", exc)
        finally:
            if img:
                try:
                    os.remove(img)
                except OSError:
                    pass
    cands.pop(src["sku"], None)
    stage(f"Ozon: checking {len(cands)} cards")

    def check(c):
        try:
            card = _card(ctx, c["sku"])
        except (_oz.OzonBlocked, _oz.OzonAgeGate):
            raise
        except Exception as exc:
            logger.warning("ozon_same card %s failed: %s", c["sku"], exc)
            return dict(c, check={"verdict": "unclear", "matched": [], "diff": [], "flags": ["карточка не открылась"]})
        full = dict(c, name=card.get("name") or c.get("name"), characteristics=card.get("characteristics"),
                    rating=c.get("rating") or card.get("rating"), reviews=c.get("reviews") or card.get("reviews"),
                    seller=c.get("seller") or card.get("seller"), price=c.get("price") or card.get("price"))
        return dict(full, check=compare(src, full))
    checked = _oz.call(ctx, "pmap", check, list(cands.values())) if cands else []
    same = [c for c in checked if c["check"]["verdict"] == "same"]
    own = dict(src, via="this card", check={"verdict": "same", "matched": [], "diff": [], "flags": []})
    return {"source": src, "same": same,
            "different": [c for c in checked if c["check"]["verdict"] == "different"],
            "unclear": [c for c in checked if c["check"]["verdict"] == "unclear"],
            "best": rank(same + [own]), "glued": bool(path)}
