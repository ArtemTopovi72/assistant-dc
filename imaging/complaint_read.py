"""What kind of complaint a picture checker made, read by the model in one call per list.

The layout critic (draw_agent.critique), the inspector (inspect_image) and the outer judge
(image_generate) each answer in free text. Word lists sorted that text before: «too dark»,
«entirely black», «missing», «wrong colour» -- and dropped «the plate text 'TE E.I76' is
slightly different from the requested 'TE Е .l 76'» as taste, a lettering mismatch shipped
(2026-10-21). bench/complaint_eval.py scores it on labelled live complaints.

kinds(ctx, texts) -> one of KINDS per text:
  dark     only about the picture or a thing in it being dark, black, a silhouette, shadowed
  blind    says the whole picture is black or dark overall
  content  names something missing, extra, doubled, wrong, misspelled, misplaced, broken
  taste    an opinion that names nothing wrong
Model down or unscripted under the suites: "content" -- the complaint stands.
"""
from __future__ import annotations
from seams import text_model

import logging
import os

logger = logging.getLogger("assistant." + __name__)

KINDS = ("dark", "blind", "content", "taste")

SYSTEM = (
    "A checker looked at a rendered picture and listed complaints about it. For each numbered "
    "complaint give its kind:\n"
    "content: it names something missing, extra or doubled, the wrong thing, the wrong colour "
    "or count of a named thing, wrong, misspelled or unreadable lettering, something misplaced, "
    "cut off, merged or broken -- also when it mentions darkness too.\n"
    "dark: it is ONLY about the picture or a thing in it being dark, black, underexposed, a "
    "silhouette, lost in shadow, or covered by black bars.\n"
    "blind: it says the whole picture is black, dark or underexposed overall.\n"
    "taste: an opinion about style, mood, neatness or composition that names nothing wrong "
    "(messy, could be more vibrant, things overlap, feels empty).\n"
    'Answer with ONE JSON object and nothing else: {"kinds": ["content", ...]}, one per '
    "complaint, in order.")


def kinds(ctx, texts) -> list:
    texts = [str(t or "") for t in (texts or [])]
    if not texts:
        return []
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        return ["content"] * len(texts)
    user = "COMPLAINTS:\n" + "\n".join(f"{i}. {t[:400]}" for i, t in enumerate(texts, 1))
    try:
        from utils import safe_json_from_llm
        raw = text_model(ctx)(ctx, SYSTEM, user, temperature=0.0, max_tokens=60 + 12 * len(texts)) or ""
        got = (safe_json_from_llm(raw, ("kinds",)) or {}).get("kinds") or []
    except Exception:
        logger.warning("complaint_read: model read failed, every complaint stands", exc_info=True)
        got = []
    out = [k if k in KINDS else "content" for k in list(got)[:len(texts)]]
    return out + ["content"] * (len(texts) - len(out))
