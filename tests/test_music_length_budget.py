"""The duration buttons must decide how long the song is.

Reported: "I set 40 seconds and it is cut off, I set three minutes and one and a
half arrives — the model must obey."

Measured on the renderer's own acoustic planner (bench/music_duration.py), which
reads MiniMaxMusic3TextEncode's `seconds` output without rendering anything:

  ceiling 60s,  9 / 17 / 33 / 65 sung lines -> planned 60.0s every time.
  ceiling 180s, 17 sung lines               -> planned 131.2s, stopped early.

So the planner sings until the WORDS run out, and only then stops; the requested
duration is a ceiling, never a target. Both reported symptoms are that one
mechanism: too few lines end the song early, too many are severed by the
ceiling.

The old prompt asked for "roughly N seconds of music", which asks the writer to
guess the renderer's singing rate. It guessed short. It now gets a line COUNT
computed from the measured rate, and the count is CHECKED rather than trusted.

Offline: no LLM, no ComfyUI, no GPU.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import music as M  # noqa: E402

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
        print("PASS  " + name)
    else:
        BAD += 1
        print("FAIL  " + name + ((": " + str(detail)) if detail else ""))
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(str(name) + ((": " + str(detail)) if detail else ""))


print("=" * 66)
print("ONLY SUNG LINES COUNT")
print("=" * 66)

LYRIC = """[intro]

[verse]
раз
два

[chorus]
три
[outro]
четыре
"""
check("section tags are structure, not words", M.count_sung_lines(LYRIC) == 4,
      M.count_sung_lines(LYRIC))
check("a lyric of nothing but tags counts zero",
      M.count_sung_lines("[intro]\n[verse]\n[outro]\n") == 0)
check("empty input does not raise", M.count_sung_lines("") == 0)
check("None does not raise", M.count_sung_lines(None) == 0)

print()
print("=" * 66)
print("THE BUDGET GROWS WITH THE SLOT")
print("=" * 66)

prev = -1
for secs in M.DURATIONS:
    target, lo, hi = M.line_budget(secs)
    check("%ss asks for a sane count" % secs, 0 < lo <= target <= hi, (target, lo, hi))
    check("%ss asks for more than the slot below it" % secs, target >= prev,
          (secs, target, prev))
    prev = target

check("no duration means no budget", M.line_budget(0) == (0, 0, 0))
check("a junk duration does not raise", M.line_budget("abc") == (0, 0, 0))
check("the count is bounded above",
      M.line_budget(10_000)[0] <= M.MAX_SUNG_LINES)

print()
print("=" * 66)
print("THE FLOOR MATCHES THE MEASUREMENT")
print("=" * 66)

# 17 lines planned 131.2s. A three-minute slot must therefore ask for MORE than
# 17, or it reproduces the exact bug that was reported.
t180, lo180, hi180 = M.line_budget(180)
check("a 180s slot asks for more than the 17 lines that gave 131s", t180 > 17,
      t180)
# Measured live, and the reason the floor was raised: a 180s slot accepted a
# 30-line lyric and the render came back at 138.3s -- 4.6s per line, not the
# 7.7s of the English pop probe the floor had been built from.
check("the 30-line lyric that produced 138s is now below the floor",
      lo180 > 30, (lo180, t180))
check("the floor, sung at the SLOWEST measured Russian rate, fills the slot",
      lo180 * M._RATE_SLOW >= 180 * 0.95, (lo180, lo180 * M._RATE_SLOW))
# The render now has headroom (render_ceiling), so the target sits near the
# middle of the measured band: sung fast the words run out a little before
# the ask, sung slow the [outro] lands inside the headroom instead of the chop.
check("the target, sung at the FASTEST measured rate, reaches most of the slot",
      t180 * M._RATE_FAST >= 180 * 0.85, (t180, t180 * M._RATE_FAST))
check("the target, sung at the SLOWEST measured rate, stays inside the render headroom",
      t180 * M._RATE_SLOW <= M.render_ceiling(180), (t180, t180 * M._RATE_SLOW, M.render_ceiling(180)))
check("the target rate is inside the measured band",
      M._RATE_FAST <= M.SECONDS_PER_SUNG_LINE <= M._RATE_SLOW, M.SECONDS_PER_SUNG_LINE)
check("and the floor sits above it in seconds per line",
      M._RATE_SLOW > M._RATE_FAST)

print()
print("=" * 66)
print("THE ASK CARRIES THE COUNT, NOT A DURATION GUESS")
print("=" * 66)

import llm as L  # noqa: E402

seen = {}


def _fake(ctx, sys_p, user_p, **kw):
    seen.setdefault("sys", sys_p)
    seen.setdefault("user", user_p)
    seen["calls"] = seen.get("calls", 0) + 1
    lines = "\n".join("строка %d" % i for i in range(1, 40))
    return ('{"lyrics": "[verse]\\n' + lines.replace("\n", "\\n")
            + '\\n[outro]\\nконец", "style": "Global Metadata\\nx\\n'
              'Vocal Details\\nfemale\\nArrangement\\nguitar"}')


_real = L.call_llm_simple
L.call_llm_simple = _fake
try:
    got = M.build_structured_caption(None, "про кота", "ru", duration_s=180)
finally:
    L.call_llm_simple = _real

check("the prompt names the line count", str(t180) in seen.get("sys", ""),
      seen.get("sys", "")[-300:])
check("the prompt no longer asks for 'roughly N seconds'",
      "roughly 180 seconds" not in seen.get("sys", ""))
check("a long enough lyric is accepted on the first attempt",
      seen.get("calls") == 1, seen.get("calls"))
check("and it comes back whole", M.count_sung_lines(got["lyrics"]) >= lo180,
      M.count_sung_lines(got["lyrics"]))

print()
print("=" * 66)
print("A SHORT LYRIC IS REWRITTEN, NOT SHIPPED")
print("=" * 66)

calls = {"n": 0, "asks": []}


def _short_then_long(ctx, sys_p, user_p, **kw):
    calls["n"] += 1
    calls["asks"].append(user_p)
    body = "\n".join("строка %d" % i for i in range(1, 4 if calls["n"] == 1 else 40))
    return ('{"lyrics": "[verse]\\n' + body.replace("\n", "\\n")
            + '\\n[outro]\\nконец", "style": "Global Metadata\\nx\\n'
              'Vocal Details\\nfemale\\nArrangement\\nguitar"}')


L.call_llm_simple = _short_then_long
try:
    got2 = M.build_structured_caption(None, "про кота", "ru", duration_s=180)
finally:
    L.call_llm_simple = _real

check("a lyric that would end the song early costs another attempt",
      calls["n"] == 2, calls["n"])
check("the retry tells the model what was wrong",
      "sung lines" in calls["asks"][1], calls["asks"][1][:160])
check("the shipped lyric is the long one",
      M.count_sung_lines(got2["lyrics"]) >= lo180,
      M.count_sung_lines(got2["lyrics"]))

# Without a duration there is nothing to enforce, and the writer must not be
# made to spend extra rungs on a rule that does not apply.
calls2 = {"n": 0}


def _always_short(ctx, sys_p, user_p, **kw):
    calls2["n"] += 1
    return ('{"lyrics": "[verse]\\nраз\\nдва\\n[outro]\\nконец", '
            '"style": "Global Metadata\\nx\\nVocal Details\\nfemale\\n'
            'Arrangement\\nguitar"}')


L.call_llm_simple = _always_short
try:
    M.build_structured_caption(None, "про кота", "ru")
finally:
    L.call_llm_simple = _real
check("with no duration asked, a short lyric is not second-guessed",
      calls2["n"] == 1, calls2["n"])

print()


import config as _config  # noqa: E402

# ── the picker must not offer what the engine cannot deliver ────────────────
# Measured end to end (bench/music_duration_e2e.py), one render each, same
# style and seed:
#
#      30s ask,   9 lines ->  30.0s  (100%, 3.33 s/line)
#      60s ask,  18 lines ->  60.0s  (100%, 3.33 s/line)
#     120s ask,  36 lines -> 120.0s  (100%, 3.33 s/line)
#     180s ask,  55 lines -> 179.9s  (100%, 3.27 s/line)
#     240s ask,  73 lines -> 149.5s   (62%, 2.05 s/line)
#     240s ask, 110 lines -> 175.6s   (73%, 1.60 s/line)
#
# The rate holds at ~3.3 s per sung line to 180 and then collapses. Every
# duration the picker offers has now been rendered and measured.
#
# Half again as many words bought 17% more song: the planner compresses as the
# lyric grows (2.05 s per line at 73, 1.60 s at 110), so the slot cannot be
# filled by writing more. An option that delivers 62% of its promise is worse
# than one that is not offered.
check("240s is no longer offered: it was measured at 62% of the ask",
      240 not in M.DURATIONS, M.DURATIONS)
check("the longest offered duration was measured as deliverable",
      max(M.DURATIONS) == 180, M.DURATIONS)
check("the clamp agrees with the picker, so a direct caller cannot ask for more",
      _config.MUSIC_MAX_SECONDS == max(M.DURATIONS),
      (_config.MUSIC_MAX_SECONDS, max(M.DURATIONS)))
check("the short end is untouched", M.DURATIONS[0] == 30, M.DURATIONS)
check("every offered duration still gets a sane budget",
      all(M.line_budget(d)[1] <= M.line_budget(d)[0] <= M.MAX_SUNG_LINES
          for d in M.DURATIONS),
      [(d, M.line_budget(d)) for d in M.DURATIONS])


print("%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)