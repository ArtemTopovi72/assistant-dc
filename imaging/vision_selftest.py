"""Does the vision model see colours, or a black picture?

2026-09-12: the house model's f16 mmproj (the vision projector LM Studio loads
beside the GGUF) turned out to be broken -- every picture came through as
"extremely dark", a black cat on a green sofa was "a dark silhouette", Cyrillic
signs were misread -- and the app grew overrules, OCR fallbacks and blind-
critic filters to live with it. A BF16 projector from the same author fixed
all of it at once. Nothing in the app would have noticed if the bad file came
back, so this asks the model about a picture whose answer is known.

Runs once at start-up, off the main thread, and only logs; a failure must
not stop the app from coming up. Skipped under F5_TEST_RUN.
"""
from __future__ import annotations

import logging
import os
import re
import threading

logger = logging.getLogger("assistant.vision_selftest")

_LAST: dict = {"ok": None, "answer": ""}


# A synthetic picture (red circle, green square on grey) PASSES on the broken
# projector -- the defect only shows on natural images, where dark greens and
# blacks collapse into "silhouettes". So the probe is a crop of one of our own
# renders: a black cat asleep on a dark green sofa, a white cup, an open book.
# Measured on the bad projector, twice: "There is no animal in this picture;
# the image consists of dark silhouettes". On the good one: "a black cat on a
# green sofa".
PROBE_IMAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "assets", "vision_selftest.jpg")

QUESTION = ("Answer in one line: is there an animal in this picture, what animal, "
            "what colour is it, and what colour is the sofa?")


def judge(answer: str) -> bool:
    a = (answer or "").lower()
    if re.search(r"\bno animal|silhouette", a):
        return False
    return bool(re.search(r"\bcat\b", a) and re.search(r"\bblack\b", a)
                and re.search(r"\bgreen\b", a))


MODELS_DIR = os.path.join(os.path.expanduser("~"), ".lmstudio", "models")
_F16_RE = re.compile(r"mmproj.*f16.*\.gguf$", re.I)
_HEALED = {"done": False}


def quarantine_f16_projectors(models_dir: str = MODELS_DIR) -> list:
    """Rename every f16 projector that sits next to a BF16 one.

    The broken f16 mmproj keeps coming back (a re-download or a model update
    puts it beside the BF16 again) and LM Studio may pick it. With the BF16 in
    the same folder the f16 is never the one we want, so it is moved aside, not
    deleted. Folders with ONLY an f16 are left alone -- that is the model's
    sole projector. Returns the renamed paths."""
    moved = []
    for root, _dirs, files in os.walk(models_dir):
        if not any(re.search(r"mmproj.*bf16.*\.gguf$", f, re.I) for f in files):
            continue
        for f in files:
            if _F16_RE.search(f) and not re.search(r"bf16", f, re.I):
                src = os.path.join(root, f)
                dst = src + ".broken-f16.bak"
                n = 2
                while os.path.exists(dst):
                    dst = f"{src}.broken-f16.bak{n}"; n += 1
                try:
                    os.replace(src, dst)
                    moved.append(dst)
                    logger.warning("moved the broken f16 projector aside: %s", dst)
                except OSError as exc:
                    logger.warning("could not move %s aside: %s", src, exc)
    return moved


def _reload_house_model(ctx) -> bool:
    """Reload the served model so LM Studio re-reads the projector."""
    try:
        import lmstudio as _lms
        from llm import LM_STUDIO_BASE
        model = getattr(ctx, "model_name", "") or os.getenv("MODEL_NAME", "")
        if not model:
            ids = _lms.loaded_model_ids(LM_STUDIO_BASE)
            model = ids[0] if ids else ""
        if not model:
            return False
        n_ctx = _lms.loaded_context_length(LM_STUDIO_BASE, model) or 20480
        ok, msg = _lms.reload_via_cli(model, n_ctx, -1, _lms.loaded_parallel(model) or 0)
        logger.warning("vision self-heal: reloaded %s (ctx %s): %s", model, n_ctx, str(msg)[:120])
        return ok
    except Exception as exc:
        logger.warning("vision self-heal: reload failed: %s", exc)
        return False


def _ask(ctx) -> str:
    from llm import analyze_image_with_llm
    return analyze_image_with_llm(ctx=ctx, image_path=PROBE_IMAGE, user_text=QUESTION,
                                  system_prompt="You describe pictures accurately.",
                                  max_tokens=600) or ""  # QAT reasons first: 80 left no answer


def run(ctx) -> bool | None:
    """Ask; on a blind answer, move a stray f16 projector aside, reload the
    model and ask again (once per process). True/False for a verdict, None
    when the call did not happen."""
    if os.getenv("F5_TEST_RUN"):
        return None
    try:
        quarantine_f16_projectors()
    except Exception as exc:
        logger.warning("projector check failed: %s", exc)
    ok = _run_once(ctx)
    if ok is False and not _HEALED["done"]:
        _HEALED["done"] = True
        quarantine_f16_projectors()
        if _reload_house_model(ctx):
            ok = _run_once(ctx)
            logger.warning("vision self-heal: after reload the model %s",
                           "SEES again" if ok else "is STILL blind")
    return ok


def _run_once(ctx) -> bool | None:
    try:
        ctx.last_llm_error = ""
    except Exception:
        pass
    try:
        answer = _ask(ctx)
    except Exception as exc:
        logger.warning("vision self-test could not run: %s", exc)
        return None
    if getattr(ctx, "last_llm_error", "") == "no_vision":
        # A text-only model is not a blind one: no projector to blame, and a
        # reload cannot give it eyes (it used to reload for 11 s and log
        # "broken projector" for muse-glimmer every start).
        logger.info("vision self-test skipped: the loaded model has no vision")
        _LAST.update(ok=None, answer="")
        return None
    ok = judge(answer)
    _LAST.update(ok=ok, answer=answer.strip())
    if ok:
        logger.info("vision self-test OK: %s", answer.strip()[:120])
    else:
        logger.error("VISION SELF-TEST FAILED -- the model does not see the picture "
                     "(answer: %r). Check the mmproj file next to the GGUF in LM Studio: "
                     "a broken projector makes every picture look black and every "
                     "judge/inspector/OCR call wrong.", answer.strip()[:160])
    return ok


def last() -> dict:
    return dict(_LAST)


def run_in_background(ctx) -> threading.Thread | None:
    if os.getenv("F5_TEST_RUN"):
        return None
    t = threading.Thread(target=run, args=(ctx,), name="vision-selftest", daemon=True)
    t.start()
    return t
