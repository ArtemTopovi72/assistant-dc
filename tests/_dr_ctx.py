"""A context for one research run with its stand-ins on it, instead of rebinding names
inside the engine's modules.

    ctx = research_ctx(FakeCtx(), plan_queries=fake, DR_MULTIHOP_ENABLED=False,
                       call_llm_simple=fake_llm, requests=fake_requests)

DR_* knobs become the run's settings (`ctx.research_settings`), `call_llm_simple`
the context's text model, everything else `ctx.deps`. A name the engine does not reach
through `dep(ctx, ...)` raises: faking it would be a silent no-op and the suite would
make the real network or LM Studio call while printing PASS.
"""
import glob
import os
import types

import dr_settings
from seams import Over

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEAMED = None


def seamed() -> set:
    """Every name the research engine reaches through `dep(ctx, "name", ...)` / `_stage`."""
    global _SEAMED
    if _SEAMED is None:
        import re
        src = "".join(open(f, encoding="utf-8").read()
                      for f in glob.glob(os.path.join(_ROOT, "research", "*.py")))
        _SEAMED = set(re.findall(r'(?:dep|_stage)\(ctx, "(\w+)"', src))
        _SEAMED |= {"collect_sources", "crawl_pages", "dedupe_pages", "rerank_for_briefing",
                    "brief_source", "synthesize_report", "plan_queries"}
    return _SEAMED


def research_ctx(ctx=None, **names):
    ctx = ctx if ctx is not None else types.SimpleNamespace(is_cancelled=lambda: False)
    knobs = {k: v for k, v in names.items() if k in dr_settings.SETTING_NAMES}
    deps = {k: v for k, v in names.items() if k not in knobs}
    if "call_llm_simple" in deps:
        ctx.text_model = deps.pop("call_llm_simple")
    unknown = set(deps) - seamed()
    if unknown:
        raise AttributeError(f"the research engine does not take {sorted(unknown)} from the "
                             f"context: faking them would change nothing")
    if knobs:
        ctx.research_settings = Over(getattr(ctx, "research_settings", None) or dr_settings, **knobs)
    if deps:
        old = vars(getattr(ctx, "deps", None) or types.SimpleNamespace())
        ctx.deps = types.SimpleNamespace(**{**old, **deps})
    return ctx
