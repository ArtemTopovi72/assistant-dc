"""Patch a research-engine seam wherever it actually lives.

The Ultra Search pipeline is split across `deep_research` + the `dr_*` stage
modules, so a name a suite wants to fake — `call_llm_simple`, `requests`,
`_acquire_page`, `_think_call` — is bound in whichever module USES it, not
necessarily in `deep_research`. Setting it on `deep_research` alone would
silently leave the real thing in force in the stage module, and the suite would
quietly make a real LM Studio or network call while still printing PASS.

`Patches` fixes that by rebinding the name in EVERY engine module that currently
binds it (and refusing to run if no module does), then restoring all of them.
The DR_* knobs are the one exception: they have a single home in `dr_settings`
and `deep_research.DR_X` is a live view on it, so patching that one name is
enough and is what happens here.

Usage mirrors the old per-suite helper:

    with Patches(call_llm_simple=lambda *a, **k: "..."):
        ...
"""
import deep_research as _DR

import dr_assemble
import dr_brief
import dr_calls
import dr_collect
import dr_crawl
import dr_policy
import dr_progress
import dr_relevance
import dr_settings
import dr_state
import dr_synthesis

#: Every module the engine is split across. Order is irrelevant — a name is
#: patched in all of them that bind it.
ENGINE_MODULES = (_DR, dr_assemble, dr_brief, dr_calls, dr_collect, dr_crawl,
                  dr_policy, dr_progress, dr_relevance, dr_state, dr_synthesis)


def homes(name):
    """The engine modules that currently bind `name` in their own __dict__."""
    if name in dr_settings.SETTING_NAMES:
        # One home; deep_research.NAME is a live view on it (see dr_settings).
        return [_DR]
    return [m for m in ENGINE_MODULES if name in vars(m)]


class Patches:
    """Rebind engine seams for the duration of a `with` block."""

    def __init__(self, **kw):
        self.kw = kw
        self.saved = []          # (module, name, original)

    def __enter__(self):
        for name, value in self.kw.items():
            targets = homes(name)
            if not targets:
                raise AttributeError(
                    f"no research-engine module binds {name!r} — patching it "
                    f"would be a silent no-op, so the suite is wrong, not the code")
            for mod in targets:
                self.saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, value)
        return self

    def __exit__(self, *exc):
        for mod, name, original in reversed(self.saved):
            setattr(mod, name, original)
        self.saved.clear()
        return False
