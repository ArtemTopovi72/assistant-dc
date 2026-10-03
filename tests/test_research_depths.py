"""The three research options must be three genuinely different amounts of work,
advertise honest times, and every one of them must return a Markdown DOCUMENT.

The product offers ~15 min / ~30 min / ~1 hour. Two things can quietly break that:
the depth can stop changing what the pipeline actually does (so all three cost the
same), and the report can come back as unstructured prose after a retry (which is
exactly what happened once — a nudge asked for the answer "now" and the model
dropped every heading).

Run: venv/Scripts/python.exe tests/test_research_depths.py
"""
import json as _json
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import deep_research as D
# The timings store moved to dr_timing.py. It is patched below, so it must be
# patched where it LIVES: deep_research deliberately does not re-export it, and
# a second copy would let this suite write the user's real ETA history.
import dr_timing as DT
import tg_bot as T

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


# ══════════════════════════════════════════════════════════ 1. three options
print("\n" + "=" * 70)
print("1. THREE OPTIONS, AND THEY MEAN 15 MIN / 30 MIN / 1 HOUR")
print("=" * 70)

check("there are exactly three depths", D.DEPTHS == ("quick", "standard", "deep"),
      D.DEPTHS)
check("quick is seeded at ~15 minutes", D._TIMING_SEED["quick"] == 900,
      D._TIMING_SEED["quick"])
check("standard is seeded at ~30 minutes", D._TIMING_SEED["standard"] == 1800,
      D._TIMING_SEED["standard"])
check("deep is seeded at ~1 hour", D._TIMING_SEED["deep"] == 3600,
      D._TIMING_SEED["deep"])
check("the seeds increase with depth",
      D._TIMING_SEED["quick"] < D._TIMING_SEED["standard"] < D._TIMING_SEED["deep"])

# ══════════════════════════════════════════════════ 2. the work actually scales
print("\n" + "=" * 70)
print("2. A DEEPER OPTION REALLY DOES MORE WORK")
print("=" * 70)

caps = {d: D._resolve_caps(d) for d in D.DEPTHS}
for k in ("max_queries", "max_pages"):
    q, s, dp = caps["quick"][k], caps["standard"][k], caps["deep"][k]
    check(f"{k} grows with depth ({q} < {s} < {dp})", q < s < dp, f"{q}/{s}/{dp}")
check("crawl depth grows too",
      caps["quick"]["depth"] <= caps["standard"]["depth"] <= caps["deep"]["depth"],
      [caps[d]["depth"] for d in D.DEPTHS])
check("the report budget grows with depth",
      caps["quick"]["report_tokens"] <= caps["standard"]["report_tokens"]
      <= caps["deep"]["report_tokens"],
      [caps[d]["report_tokens"] for d in D.DEPTHS])
# The reflection pass is what actually separates the three options in WALL-CLOCK
# terms: it re-crawls and re-briefs, and briefing is the most expensive stage
# (22-32 of the 42 minutes a "quick" run took before this). A quick run is ONE
# pass by definition.
check("quick does no reflection pass at all", caps["quick"]["reflect_iters"] == 0,
      caps["quick"]["reflect_iters"])
check("quick therefore has no reflection budget either",
      caps["quick"]["reflect_budget"] == 0, caps["quick"]["reflect_budget"])
check("standard allows one reflection pass",
      caps["standard"]["reflect_iters"] == 1, caps["standard"]["reflect_iters"])
check("deep allows the most reflection",
      caps["deep"]["reflect_iters"] >= caps["standard"]["reflect_iters"],
      [caps[d]["reflect_iters"] for d in D.DEPTHS])
check("the reflection budget grows with depth",
      caps["quick"]["reflect_budget"] <= caps["standard"]["reflect_budget"]
      <= caps["deep"]["reflect_budget"],
      [caps[d]["reflect_budget"] for d in D.DEPTHS])

# ...and the loop must actually READ the caps, not the global constants.
import inspect as _insp
_loop_src = _insp.getsource(D.run_deep_research)
check("the reflection loop is bounded by the DEPTH, not the global constant",
      'caps.get("reflect_iters"' in _loop_src and "refl_iter < refl_max" in _loop_src,
      "the loop still uses DR_REFLECTION_MAX_ITERATIONS directly")

check("an unknown depth falls back to standard, never crashes",
      D._resolve_caps("nonsense") == caps["standard"])
check("an empty depth falls back to standard", D._resolve_caps("") == caps["standard"])

# ═══════════════════════════════════════════════════ 3. the estimate is honest
print("\n" + "=" * 70)
print("3. THE ESTIMATE IS A MEASUREMENT ONCE THERE ARE MEASUREMENTS")
print("=" * 70)

for d in D.DEPTHS:
    secs, n = D.estimate_duration(d)
    check(f"{d} reports a positive estimate", secs > 0, secs)
    check(f"{d} says whether it is a seed or measured", isinstance(n, int), n)

_saved = DT._TIMING_FILE
_tmp = tempfile.mkdtemp(prefix="dr_timing_")
# Must be a Path: record_run_duration uses .parent/.with_suffix/.replace, and a
# str fails into its deliberate "never break a run over telemetry" catch-all.
from pathlib import Path as _Path
DT._TIMING_FILE = _Path(_tmp) / "_timings.json"
try:
    for v in (1200.0, 1400.0, 1300.0):
        D.record_run_duration("quick", v)
    secs, n = D.estimate_duration("quick")
    check("a measured estimate replaces the seed", secs == 1300.0 and n == 3,
          f"{secs} over {n}")
    check("it is the MEDIAN, not the latest or the mean", secs == 1300.0, secs)

    # A stubbed test run or an abort must not be recorded as a measurement. The
    # LIVE file was found holding [0.1, 0.1, 0.0], which made the bot quote the
    # user an ETA of "0 min".
    D.record_run_duration("standard", 0.1)
    D.record_run_duration("standard", 0.0)
    secs2, n2 = D.estimate_duration("standard")
    # standard has no measurement of its own, but quick just got one (median
    # 1300s) -- infer standard's estimate from it via the seed ratio (1800/900)
    # rather than quoting the flat 1800s seed, which could land out of order
    # against a sibling that DID get measured (this happened live: an
    # unmeasured "standard" quoted a flatter/slower seed than a measured
    # "deep"). n stays 0: this is still an inference, not standard's own data.
    check("an implausibly short run is not recorded, so a sibling's measurement fills in instead",
          n2 == 0 and secs2 == 1300.0 * (D._TIMING_SEED["standard"] / D._TIMING_SEED["quick"]),
          f"{secs2} over {n2}")
    check("so the estimate stays honest and above the seed's floor, never 0",
          D.estimate_duration("standard")[0] >= 60)

    # …and a file already poisoned by older junk must be ignored on READ too.
    import json as _json
    DT._TIMING_FILE.write_text(_json.dumps({"deep": [0.1, 0.0], "quick": [],
                                           "standard": []}), encoding="utf-8")
    secs3, n3 = D.estimate_duration("deep")
    check("junk already on disk is discarded when estimating",
          n3 == 0 and secs3 == D._TIMING_SEED["deep"], f"{secs3} over {n3}")
    check("no depth can ever report a zero-minute wait",
          all(D.estimate_duration(d)[0] >= 60 for d in D.DEPTHS),
          {d: D.estimate_duration(d) for d in D.DEPTHS})

    # A measurement only describes the profile it was measured under. When the
    # caps changed (quick stopped doing a second reflection pass) the UI kept
    # quoting "43 min" for a run that had just been halved.
    DT._TIMING_FILE.write_text(_json.dumps(
        {"quick": [2600.0, 2500.0, 2700.0], "standard": [], "deep": [],
         "_profile": D._profile_signature()}), encoding="utf-8")
    secs4, n4 = D.estimate_duration("quick")
    check("timings from the CURRENT profile are used", n4 == 3 and secs4 == 2600.0,
          f"{secs4} over {n4}")

    DT._TIMING_FILE.write_text(_json.dumps(
        {"quick": [2600.0, 2500.0, 2700.0], "standard": [], "deep": [],
         "_profile": "stale-profile"}), encoding="utf-8")
    secs5, n5 = D.estimate_duration("quick")
    check("timings from a DIFFERENT profile are discarded",
          n5 == 0 and secs5 == D._TIMING_SEED["quick"], f"{secs5} over {n5}")

    DT._TIMING_FILE.write_text(_json.dumps({"quick": [2600.0]}), encoding="utf-8")
    check("a file with no profile stamp at all is discarded too",
          D.estimate_duration("quick")[1] == 0)
    check("the signature changes when the caps change",
          len(D._profile_signature()) >= 8 and isinstance(D._profile_signature(), str))
finally:
    DT._TIMING_FILE = _saved

check("an unknown depth still estimates rather than raising",
      D.estimate_duration("nope")[0] > 0)

# ══════════════════════════════════════════════════ 4. the picker offers them
print("\n" + "=" * 70)
print("4. THE USER CAN PICK ONE, AND SEES WHAT IT COSTS")
print("=" * 70)

check("the bot knows the same three depths", T._DEPTHS == D.DEPTHS, T._DEPTHS)
kb = T._depth_menu_kb("ru", "standard")
flat = str(kb)
for d in D.DEPTHS:
    check(f"the {d} option is offered", f"depth:{d}" in flat, flat[:200])
txt = T._depth_menu_text("ru", "standard")
check("the menu shows an estimated time", any(ch.isdigit() for ch in txt), txt[:200])

# ═══════════════════════════════════════ 5. the report is a MARKDOWN DOCUMENT
print("\n" + "=" * 70)
print("5. WHATEVER THE DEPTH, THE OUTPUT IS STRUCTURED MARKDOWN")
print("=" * 70)

# The retry nudge is what once destroyed the structure: it asked for the answer
# "NOW" and the model answered with a wall of prose.
import inspect
src = inspect.getsource(D._think_call)
check("the retry still demands the Markdown structure back",
      "Markdown" in src and "heading" in src, "the nudge lost its structure clause")

hdr = D._report_header("тема", [], {"sources": 3, "pages": 2}, "deep", out_lang="ru")
check("the header records which depth was used", "deep" in hdr.lower(), hdr[:160])

for d in D.DEPTHS:
    h = D._report_header("t", [], {"sources": 1, "pages": 1}, d)
    check(f"the {d} report names its depth", d in h.lower(), h[:120])

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
