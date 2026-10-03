"""Lettering that sits ON something stays on it.

Live, 2026-09-12: the planner put "Для гостей театра на Пролетарской" on the
champagne label, and the read-back loop's single-line rule resized that box to
0.94 x 0.047 -- a strip across the whole frame -- because 34 characters at the
glyph floor need 1.6 frames of width. The label floated off its bottle and the
words were painted as a banner across the picture. Rendered with the fix, same
seed, the label reads correctly in three lines (runtime/label_wrap_test.png).

Pure geometry; nothing here touches a model or the GPU.

Run: venv/Scripts/python.exe tests/test_lettering_stays_on_host.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import logging; logging.basicConfig(level=logging.CRITICAL)

import draw_text as D
import text_layout as T

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


def inside(a, b, tol=0.005):
    return (a["x"] >= b["x"] - tol and a["y"] >= b["y"] - tol
            and a["x"] + a["w"] <= b["x"] + b["w"] + tol
            and a["y"] + a["h"] <= b["y"] + b["h"] + tol)


LABEL = "Для гостей театра на Пролетарской"
LIVE = {"elements": [
    {"desc": "a luxury champagne bottle with ornate details", "text": "",
     "x": 0.3, "y": 0.15, "w": 0.4, "h": 0.75},
    {"desc": "an ornate label on the bottle", "text": LABEL,
     "x": 0.35, "y": 0.55, "w": 0.30, "h": 0.10},
    {"desc": "a dark, reflective elegant surface", "text": "",
     "x": 0.0, "y": 0.8, "w": 1.0, "h": 0.2},
]}

# ── wrapping ─────────────────────────────────────────────────────────────────
check("balanced wrap, not greedy",
      T.wrap_text(LABEL, 18) == ["Для гостей театра", "на Пролетарской"], T.wrap_text(LABEL, 18))
check("a word is never split",
      all(" " not in ln or len(ln) <= 12 for ln in T.wrap_text(LABEL, 12)), T.wrap_text(LABEL, 12))
check("a short string is one line", T.wrap_text("CAFE", 18) == ["CAFE"])
check("lines_of reads a wrapped string back", T.lines_of("a\nb\n c ") == ["a", "b", "c"])

wrapped, w, h = T.fit_text_in(LABEL, 0.4 * 0.92, 0.047)
check("the label wraps into at most three lines", len(T.lines_of(wrapped)) <= T.MAX_TEXT_LINES, wrapped)
check("every line is at the per-line floor or above",
      h / len(T.lines_of(wrapped)) >= T.MIN_LINE_H - 1e-9, h)

# ── the live case ────────────────────────────────────────────────────────────
out, notes = D.auto_fix_text(copy.deepcopy(LIVE))
bottle, label = out["elements"][0], out["elements"][1]
check("the label is still inside the bottle", inside(label, bottle), (label, bottle))
check("and not a strip across the frame", label["w"] < 0.7, label["w"])
check("the text is wrapped", "\n" in label["text"], label["text"])
check("nothing was lost in the wrap",
      T._norm_text(label["text"]) == T._norm_text(LABEL))
# At the per-line floor the label fits this bottle without a closer shot;
# the shot moves in only when it must (checked on a narrower host below).
check("the bottle was not enlarged when the label already fits",
      abs(bottle["w"] - 0.4) < 0.02 and not any("moved in" in n for n in notes), notes)
narrow = copy.deepcopy(LIVE); narrow["elements"][0].update({"x": 0.4, "w": 0.2})
narrow["elements"][1].update({"x": 0.42, "w": 0.16})
o_n, n_n = D.auto_fix_text(narrow)
check("a host too narrow for its lettering is moved in on",
      o_n["elements"][0]["w"] > 0.2 and any("moved in" in n for n in n_n), n_n)
check("the bottle still fits the frame",
      bottle["x"] >= 0 and bottle["x"] + bottle["w"] <= 1 and bottle["y"] + bottle["h"] <= 1, bottle)
check("the report has nothing to complain about", D.text_report(out) == [], D.text_report(out))
check("a second pass is a no-op", D.auto_fix_text(out)[1] == [], D.auto_fix_text(out)[1])

# ── what must not change ─────────────────────────────────────────────────────
banner = {"elements": [
    {"desc": "a street", "text": "", "x": 0, "y": 0, "w": 1, "h": 1},
    {"desc": "a banner", "text": "WELCOME TO THE FESTIVAL", "x": 0.1, "y": 0.1, "w": 0.8, "h": 0.1}]}
o2, _ = D.auto_fix_text(banner)
check("lettering on the background is a banner and may be wide",
      o2["elements"][1]["w"] > 0.9 and "\n" not in o2["elements"][1]["text"], o2["elements"][1])

sign = {"elements": [
    {"desc": "a shop front", "text": "", "x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6},
    {"desc": "the sign", "text": "CAFE", "x": 0.35, "y": 0.25, "w": 0.3, "h": 0.1}]}
o3, _ = D.auto_fix_text(sign)
s_el, host = o3["elements"][1], o3["elements"][0]
check("a short word on a sign is sized the old way and stays put",
      s_el["text"] == "CAFE" and inside(s_el, host)
      and abs(host["w"] - 0.6) < 1e-9, (s_el, host))

# The host is the SMALLEST element under the text's centre: the label lives on
# the bottle, not on the table the bottle stands on.
stack = {"elements": [
    {"desc": "a table", "text": "", "x": 0.0, "y": 0.3, "w": 1.0, "h": 0.5},
    {"desc": "a bottle", "text": "", "x": 0.4, "y": 0.35, "w": 0.2, "h": 0.4},
    {"desc": "label", "text": LABEL, "x": 0.42, "y": 0.5, "w": 0.16, "h": 0.06}]}
check("the host is the smallest element under the text",
      D._host_of(stack, 2)["desc"] == "a bottle")

# Line-aware report: a wrapped box is judged per line.
rep = D.text_report({"elements": [
    {"desc": "label", "text": "Для гостей\nтеатра на\nПролетарской",
     "x": 0.3, "y": 0.4, "w": 0.47, "h": 0.24}]})
check("a three-line box is not flagged as the wrong shape or too small", rep == [], rep)

# -- what the four agent runs of 2026-09-12 taught -----------------------------
import draw_agent as DA

# 1. the critic may not delete the requested lettering, nor what it is on
lay = {"elements": [
    {"desc": "bottle", "text": "", "x": 0.3, "y": 0.1, "w": 0.4, "h": 0.8},
    {"desc": "label", "text": LABEL, "x": 0.35, "y": 0.4, "w": 0.3, "h": 0.2},
    {"desc": "a chair", "text": "", "x": 0.0, "y": 0.5, "w": 0.2, "h": 0.3}]}
o, n = DA.apply_ops(lay, [{"op": "delete", "target": "the label"}])
check("the critic cannot delete the requested lettering", len(o["elements"]) == 3, n)
o, n = DA.apply_ops(lay, [{"op": "delete", "target": "the bottle"}])
check("nor the thing it is written on", len(o["elements"]) == 3, n)
o, n = DA.apply_ops(lay, [{"op": "delete", "target": "the chair"}])
check("anything else it may still delete", len(o["elements"]) == 2, n)
o, n = DA.apply_ops(lay, [{"op": "delete", "target": "the label"}], instruction="убери надпись")
check("the user's own 'remove the lettering' goes through", len(o["elements"]) == 2, n)

# 2. multi-word lettering is restated line by line, never hyphenated
el = {"desc": "the main label", "text": "Для гостей\nтеатра на\nПролетарской"}
sp = D.spell_out(el)
check("no letter-by-letter hyphens for a phrase (the model painted them)",
      "-л-я" not in sp and "“Для гостей” / “театра на”" in sp, sp)
el["desc"] = sp
check("spell_out is idempotent", D.spell_out(el) == sp)
check("a single word gets its letter count, not hyphens (painted as ВЕ-Ч-ЕР)",
      "6-letter word" in D.spell_out({"desc": "a door", "text": "POLICE"}) and "-O-" not in D.spell_out({"desc": "a door", "text": "POLICE"}))

# 3. the read-back matches a wrapped label line by line
check("three lines read as three strings match the whole",
      D._match_lines(LABEL.replace(" театра", "\nтеатра").replace(" на ", " на\n"),
                     ["Для гостей", "театра на", "Пролетарской"])[1] >= 0.99)
check("one line alone does not pass for the whole",
      D._match_lines("Для гостей\nтеатра на\nПролетарской", ["Пролетарской"])[1] < 0.8)

# 4. stray lettering is a failure and its suppression is idempotent
lay2, notes2 = D.suppress_stray_text(copy.deepcopy(LIVE), ["ПОМАPIA"])
check("suppression names the parts of the host that must be blank",
      "plain and unmarked" in lay2["elements"][0]["desc"])
check("and says the label itself is wanted (so the critic does not delete it)",
      "must be there" in lay2["high_level_description"])
check("it is detected as already applied", D.stray_suppressed(lay2))
lay3, _ = D.suppress_stray_text(lay2, ["X"])
check("applying it twice does not double the wording",
      lay3["high_level_description"].count("one and only") == 1)
check("stray text is listed as a problem",
      any("nobody asked for" in p for p in D.text_problems([], ["ПОМАPIA"])))

# 5. headroom: the space above a lettered host becomes an element
o, n = D.auto_fix_text(copy.deepcopy(LIVE))
check("the strip above the bottle is now an element of its own",
      any(D.HEADROOM_NOTE in e["desc"] for e in o["elements"]), [e["desc"][:30] for e in o["elements"]])
check("appended, so the other elements keep their places",
      o["elements"][0]["desc"].startswith("a luxury champagne bottle"))
check("no headroom for a host that starts at the top",
      not any(D.HEADROOM_NOTE in e["desc"] for e in D.auto_fix_text(
          {"elements": [{"desc": "a tower", "text": "", "x": 0.3, "y": 0.0, "w": 0.4, "h": 0.9},
                        {"desc": "sign", "text": "HOTEL", "x": 0.4, "y": 0.4, "w": 0.2, "h": 0.08}]})[0]["elements"]))

# 6. the per-line floor for wrapped text
check("wrapped lines may sit at MIN_LINE_H, not MIN_TEXT_H",
      T.MIN_LINE_H < T.MIN_TEXT_H and T.fit_text_in(LABEL, 0.35, 0.05)[2] / 3 <= T.MIN_LINE_H + 1e-9)


print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
