"""The deep-research runtime knobs have exactly ONE home.

Every DR_* knob the Ultra Search pipeline reads lives in `dr_settings`, and the
pipeline modules read them as attributes (`S.DR_MAX_PAGES`) so a runtime change
is visible everywhere at once. Two things make that fragile enough to be worth
pinning down:

  1. The historical patch surface is `deep_research.DR_X` — the GUI's Manual
     Control panel seeds its widgets from it, and half a dozen suites patch it
     directly. `deep_research` keeps that surface alive with a module proxy that
     forwards reads AND writes to `dr_settings`. If someone ever re-adds a
     `from config import DR_X` (or a `from dr_settings import DR_X`) to
     deep_research, that binding lands in the module __dict__, shadows the
     proxy, and silently becomes a SECOND copy: the panel would show the new
     value while the pipeline kept running on the old one. The tests below fail
     loudly on that.

  2. `apply_overrides`/`restore_overrides` must round-trip through the same home,
     and must validate everything before mutating anything (a half-applied set
     leaks permanently, since the caller never receives `saved`).

Pure in-process attribute checks — no LM Studio, no network, no GPU.

Run: venv/Scripts/python.exe tests/test_dr_settings.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import ast

import deep_research as D
import dr_settings as S

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}")
        if detail:
            print(f"       {detail}")


# ─────────────────────────────────────────────── (a) one home, not two copies
print("ONE HOME")

check("dr_settings names every knob it owns",
      "DR_MAX_PAGES" in S.SETTING_NAMES and "DR_SURVEY_MODE" in S.SETTING_NAMES
      and "LM_STUDIO_BASE" in S.SETTING_NAMES)

check("every manual-control knob is a name dr_settings actually owns",
      set(S.MANUAL_OVERRIDE_SPEC) <= set(S.SETTING_NAMES),
      sorted(set(S.MANUAL_OVERRIDE_SPEC) - set(S.SETTING_NAMES)))

shadowed = sorted(n for n in S.SETTING_NAMES if n in vars(D))
check("no knob is re-bound in deep_research's own __dict__ (that would shadow "
      "the proxy and create a stale second copy)",
      not shadowed, shadowed)

# The static form of the same rule: catch a re-introduced `from config import
# DR_...` even if the module happened not to be imported in this process.
_src = ast.parse(open(os.path.join(os.path.dirname(__file__), "..", "research/deep_research.py"),
                      encoding="utf-8").read())
_imported = set()
for _n_ in ast.walk(_src):
    if isinstance(_n_, ast.ImportFrom):
        for a in _n_.names:
            _imported.add(a.asname or a.name)
check("deep_research does not `from`-import any knob (only `import dr_settings as S`)",
      not (_imported & set(S.SETTING_NAMES)),
      sorted(_imported & set(S.SETTING_NAMES)))


# ───────────────────────────────────────────── (b) the proxy is a live view
print("\nTHE deep_research.DR_X PROXY IS A VIEW, NOT A COPY")

_orig = S.DR_MAX_SOURCES
try:
    check("a read on deep_research forwards to dr_settings",
          D.DR_MAX_SOURCES == S.DR_MAX_SOURCES)

    D.DR_MAX_SOURCES = _orig + 41           # what the suites do
    check("a write through deep_research lands in dr_settings (so every "
          "pipeline module sees it)",
          S.DR_MAX_SOURCES == _orig + 41, S.DR_MAX_SOURCES)
    check("and reads back through deep_research", D.DR_MAX_SOURCES == _orig + 41)
finally:
    D.DR_MAX_SOURCES = _orig

check("restoring through the proxy restores the real home",
      S.DR_MAX_SOURCES == _orig)

check("a non-knob attribute still resolves normally",
      callable(D.run_deep_research))

_raised = False
try:
    D._TIMING_FILE                          # deliberately NOT re-exported here
except AttributeError:
    _raised = True
check("an unknown attribute still raises AttributeError (the proxy does not "
      "swallow typos)", _raised)


# ───────────────────────────────────────── (c) overrides round-trip one home
print("\nMANUAL OVERRIDES ROUND-TRIP THROUGH THAT ONE HOME")

_before = (S.DR_MAX_PAGES, S.DR_SURVEY_MODE)
saved = D.apply_overrides({"DR_MAX_PAGES": 5, "DR_SURVEY_MODE": False})
check("apply_overrides patches dr_settings", S.DR_MAX_PAGES == 5)
check("and the value is visible on the deep_research surface too",
      D.DR_MAX_PAGES == 5)
check("_resolve_caps reads the OVERRIDDEN value, not an import-time snapshot",
      D._resolve_caps("standard")["max_pages"] == 5,
      D._resolve_caps("standard"))
D.restore_overrides(saved)
check("restore_overrides puts the originals back",
      (S.DR_MAX_PAGES, S.DR_SURVEY_MODE) == _before,
      (S.DR_MAX_PAGES, S.DR_SURVEY_MODE))

_raised = False
try:
    D.apply_overrides({"DR_NOT_A_KNOB": 1})
except KeyError:
    _raised = True
check("an unknown knob raises rather than being silently ignored", _raised)

_before_pages = S.DR_MAX_PAGES
_raised = False
try:
    # valid knob first, invalid second — nothing may be applied
    D.apply_overrides({"DR_MAX_PAGES": 5, "DR_MAX_QUERIES": 9999})
except ValueError:
    _raised = True
check("an out-of-range value raises", _raised)
check("and nothing was half-applied (validate-all-then-mutate)",
      S.DR_MAX_PAGES == _before_pages, S.DR_MAX_PAGES)

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
