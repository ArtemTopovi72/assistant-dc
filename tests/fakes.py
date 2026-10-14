"""Stand-ins handed to the code under test, never written over a module.

`Over(base, **overrides)` answers the overrides and falls through to `base` for the rest,
so a suite gets the real `video` module with a fake renderer in one object and nothing
global changes: the next suite in the same process sees the real module.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from seams import Over  # noqa: E402,F401


def posting(post, ctx=None):
    """`ctx` (a bare namespace when None) handing the code under test an http client whose
    `post` is `post`; everything else on it is the real requests."""
    import types
    import requests
    ctx = ctx if ctx is not None else types.SimpleNamespace()
    deps = getattr(ctx, "deps", None) or types.SimpleNamespace()
    deps.requests = Over(requests, post=post)
    ctx.deps = deps
    return ctx


def model_selector(models, gb=3.0):
    """The model_selector a SettingsDialog lists models from (`modules={"model_selector": ...}`):
    the real module answering `models` with every file `gb` GB. Set `.list_models` on it to
    change the list between populates."""
    import model_selector as real
    return Over(real, list_models=lambda base: models,
                _index_model_sizes=lambda d: {}, _size_gb=lambda mid, sizes: gb)


def deps_ctx(**deps):
    """A context that hands the code under test these stand-ins (core/seams.py `dep`)."""
    import types
    return types.SimpleNamespace(deps=types.SimpleNamespace(**deps))


def reads(read=None, yes=None, pick=None):
    """The model's reads a suite scripts, as the seams core/seams.py looks up on a context:
    `read(text) -> dict | None` (merged over intent.FALLBACK), `yes(question, text) -> bool`,
    `pick(question, text) -> option | None`. Unscripted: what the offline reads answer."""
    import intent
    out = {}
    if read is not None:
        def intent_read(ctx, text, previous="", attached=""):
            got = read((text or "").strip()) if (text or "").strip() else None
            return {**intent.FALLBACK, **got, "ok": True} if got else dict(intent.FALLBACK)
        out["intent_read"] = intent_read
    if yes is not None:
        def yes_no(question, text, default=False, wait=True):
            text = (text or "").strip()
            return bool(yes(question, text)) if text else default
        out["yes_no"] = yes_no
    if pick is not None:
        def choice(question, text, options, default=""):
            text = (text or "").strip()
            return (pick(question, text) or default) if text else default
        out["choice"] = choice
    return out
