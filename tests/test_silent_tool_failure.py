"""Regression: a tool that fails QUIETLY must not become "Готово!".

bench/tc_chaos.py breaks tools on purpose and scores only what the user is
finally told. The loop classified a result as failed if and only if it began
with "[TOOL ERROR]", so polite failures were handled well and silent ones were
not: against tools returning "" or a truncated string, the agent said
"Готово! Я добавил ей очки" although the tool had produced nothing at all.

A model cannot notice an absence. It has to be shown one, which is what
_detect_silent_failure does -- with two signals that are cheap and certain:

  * a blank result is never a real answer;
  * a tool whose job is to produce a file, that finished with no path in state,
    produced no file, whatever its prose claims. Those state keys are the ones
    tg_bot and the GUI actually read to deliver the artifact, so an unset key
    means the user receives nothing.

Deliberately NOT attempted here: plausible-but-wrong output with the artifact
present. That is undetectable by inspection of the result, needs the vision
verifier, and is left as the known remaining gap (chaos mode "wrong").

Run: venv/Scripts/python.exe tests/test_silent_tool_failure.py
"""
import inspect
import sys
sys.path.insert(0, ".")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import graph_personality as P
import graph_finalize as GF

checks = []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if not ok else ""))


def is_err(s):
    return s.lstrip().startswith("[TOOL ERROR]")


D = P._detect_silent_failure

# --- blank results ---------------------------------------------------------
for label, blank in [("empty", ""), ("spaces", "   "), ("newlines", "\n\n\t")]:
    out = D("search", blank, {})
    check(f"blank_{label}_becomes_an_error", is_err(out), repr(out[:60]))
check("the blank error names the tool", "search" in D("search", "", {}))
check("the blank error forbids claiming success",
      "did NOT" in D("search", "", {}) or "not" in D("search", "", {}).lower())

# --- the artifact contract -------------------------------------------------
check("generate_image with no image_path is a failure",
      is_err(D("generate_image", "Image generated successfully.", {})))
check("generate_image WITH an image_path is fine",
      not is_err(D("generate_image", "Image generated successfully.",
                   {"image_path": "outputs/x.png"})))
check("create_presentation with no document_path is a failure",
      is_err(D("create_presentation", "Deck built and sent.", {})))
check("create_presentation WITH a document_path is fine",
      not is_err(D("create_presentation", "Deck built and sent.",
                   {"document_path": "outputs/d.pptx"})))
check("generate_video with no video_path is a failure",
      is_err(D("generate_video", "Video ready (6s).", {})))
check("a whitespace-only path does not count as delivered",
      is_err(D("generate_image", "done", {"image_path": "   "})))

# Every artifact-bearing tool must be covered, or a new one silently opts out.
check("all image-producing tools are in the contract table",
      {"generate_image", "find_photo", "inpaint_image", "redraw_image",
       "transfer_image", "fix_hands", "fix_artifact"} <= set(P._TOOL_ARTIFACT),
      sorted(P._TOOL_ARTIFACT))
check("the artifact keys match what the delivery layers read",
      set(P._TOOL_ARTIFACT.values()) == {"image_path", "video_path", "document_path"},
      sorted(set(P._TOOL_ARTIFACT.values())))

# --- tools with no artifact must be left alone -----------------------------
# calculate is deliberately absent: it has its own rule (a result with no digit
# in it is not a calculation) and is checked separately below.
for tool in ("search", "remember_fact", "read_clipboard",
             "inspect_image", "deep_research"):
    check(f"{tool}_result_passes_through",
          D(tool, "a perfectly ordinary result", {}) == "a perfectly ordinary result")

# --- an existing error is not rewritten ------------------------------------
_err = "[TOOL ERROR] the backend is down."
check("an existing [TOOL ERROR] is returned unchanged",
      D("generate_image", _err, {}) == _err)
check("...even when the artifact is missing too",
      D("create_presentation", _err, {}) == _err)

# --- the verifier must be believed when it contradicts the tool ------------
# Second false-success family from the chaos bench, and a nastier one: the tool
# reports success, inspect_image looks at the picture and says the element is
# NOT there, the agent retries, is told NO again -- and then writes "Вот твой
# рыжий кот-астронавт!". It believes its verifier when deciding to retry and
# forgets it when writing the answer.
NEGATIVE = [
    ("english", "Inspection: the requested element is NOT present in the image."),
    ("unchanged", "The picture is unchanged from before the call."),
    ("missing", "the hat is missing and the hands are distorted"),
    ("no_effect", "the edit did not take effect"),
    ("russian", "запрошенный объект отсутствует на изображении"),
]
for name, text in NEGATIVE:
    check(f"verdict_negative_{name}", bool(P._VERIFY_NEGATIVE_RE.search(text)),
          repr(text[:50]))

POSITIVE = [
    ("all_present", "Inspection: subject present; requested object PRESENT; "
                    "hands look correct; no visible artifacts."),
    ("ru_ok", "объект присутствует, всё на месте"),
    # "dismissing" contains "missing"; word boundaries have to hold or every
    # clean inspection that uses the word reads as a failure.
    ("substring", "dismissing the earlier concern, the render is correct"),
]
for name, text in POSITIVE:
    check(f"verdict_positive_{name}", not P._VERIFY_NEGATIVE_RE.search(text),
          repr(text[:50]))

# Which drafts claim a result («Вот твой рыжий кот-астронавт!», «Презентация готова»,
# «Результат вычисления … равен 20 965 334.4») is the model's reading (agent/reply_read.py),
# scored live with these phrases in bench/reply_read_set. Here: the decisions made on it.
import types as _types
from fakes import claims as _claims
_C = _types.SimpleNamespace(reply_read=_claims({
    "Вот ваш рыжий кот в скафандре!": {"delivers": True}, "Я добавил ей очки.": {"did": ["draw"]},
    "К сожалению, не удалось создать изображение.": {"admits_failure": True},
    "Вот ваш рыжий кот!": {"delivers": True}, "Презентация готова.": {"did": ["file"]},
    "Результат вычисления равен 20963112": {"computed": True},
    "К сожалению, не получилось.": {"admits_failure": True}}))

# The decision itself, not a grep of the loop's source: a substring check on
# personality_node would still pass with "False and" in front of the whole
# condition. V() is the real predicate the loop calls.
V = lambda draft, neg, sent: P._answer_contradicts_reality(draft, neg, set(), sent, _C)
check("a delivery claim after a negative verdict is caught",
      V("Вот ваш рыжий кот в скафандре!", True, False))
check("a first-person claim after a negative verdict is caught",
      V("Я добавил ей очки.", True, False))
check("a clean verdict lets the same sentence through",
      not V("Вот ваш рыжий кот в скафандре!", False, False))
check("an honest answer is never caught",
      not V("К сожалению, не удалось создать изображение.", True, False))
check("the guard fires only once per turn",
      not V("Вот ваш рыжий кот!", True, True))
check("an empty draft is not a claim", not V("", True, False))

# A broken promise is a trigger on its own — no verification needed.
A = lambda draft, neg, failed, sent: P._answer_contradicts_reality(draft, neg, failed, sent, _C)
check("a failed artifact tool + a promise claim is caught",
      A("Презентация готова.", False, {"create_presentation"}, False))
check("a failed calculate + an invented number is caught",
      A("Результат вычисления равен 20963112", False, {"calculate"}, False))
check("no failure and no negative verdict lets it through",
      not A("Презентация готова.", False, set(), False))
check("a failed tool alone is not enough without a claim",
      not A("К сожалению, не получилось.", False, {"create_presentation"}, False))

# calculate must return the VALUE, the way tools._handle_calculate does
# (str(result)). Checking merely for a digit is not enough -- chaos mode
# "wrong" hands back a numbered list, which has digits. Parsing the result as a
# number is the handler's own contract.
for label, val in [("plain", "20709273.6"), ("negative", "-42"),
                   ("scientific", "1e10"), ("bool", "True"),
                   ("inf", "inf"), ("nan", "nan"), ("complex", "(3+4j)")]:
    check(f"calc_result_{label}_passes", not is_err(D("calculate", val, {})), val)

for label, val in [("prose", "Beekeeping in temperate climates: hive placement."),
                   ("numbered_list", "1. Beekeeping guide. 2. Ten easy recipes."),
                   ("truncated_web", "[Untrusted web data] 1. The population of Tok"),
                   ("placeholder", "ok")]:
    check(f"calc_result_{label}_is_an_error", is_err(D("calculate", val, {})), val[:40])

check("only calculate is held to that rule",
      not is_err(D("search", "no numbers here at all", {})))

# --- the deterministic replacement at finalisation -------------------------
# Chosen over another corrective round: a corrective round is advice the model
# is free to ignore, a deterministic scrub is not.
# --- the same backstop for the tools that hand over a FILE -----------------
# The corrective round asks the model not to claim a result that does not
# exist; it can still claim one ("Вот реальное фото Эйфелевой башни" after
# find_photo failed). This is the deterministic net under it.
check("a failed find_photo produces an honest line",
      "фотографию" in GF._honest_failure_line({"find_photo"}))
check("several failures are named together",
      "презентацию" in GF._honest_failure_line({"create_presentation",
                                                "generate_image"})
      and "картинку" in GF._honest_failure_line({"create_presentation",
                                                 "generate_image"}))
check("a tool that hands over no file gets no line",
      GF._honest_failure_line({"search"}) == "")
# (the honest lines claim nothing: scored with the other canned answers in bench/reply_read_set)
check("every artifact tool has a noun",
      set(GF._PROMISE_NOUN) == set(P._TOOL_ARTIFACT),
      f"{sorted(set(P._TOOL_ARTIFACT) - set(GF._PROMISE_NOUN))} missing")

_fin_src = inspect.getsource(GF._finalize_answer)
check("the file backstop is wired",
      "_honest_failure_line(_artifact_fails, _tool_said_unsupported(messages))" in _fin_src
      and "final_answer = _line" in _fin_src)
check("the replacement is wired with both triggers",
      '(_degenerate or said()["computed"])' in _fin_src)
# The ASSIGNMENT, not just the condition. Checking the condition alone let a
# mutation that deleted "final_answer = _NO_CALC_ANSWER" pass unnoticed -- the
# guard would have evaluated and then done nothing.
check("...and actually replaces the answer",
      "final_answer = _NO_CALC_ANSWER" in _fin_src)

# --- fabricated image embeds -----------------------------------------------
# Pictures reach the user through state["image_path"], never a URL in the text,
# so an ![...](http…) the model wrote is always invented. It was producing
# wikimedia links when find_photo failed — worse than an empty answer, because
# it looks like it worked.
_md = ("Вот реальное фото:\n\n"
       "![Eiffel](https://upload.wikimedia.org/wikipedia/commons/a/a8/x.jpg)\n\n"
       "Надеюсь подойдёт.")
_clean = GF._strip_fabricated_images(_md)
check("the fabricated embed is removed", "![" not in _clean and "wikimedia" not in _clean,
      repr(_clean))
check("the surrounding text survives",
      "Вот реальное фото" in _clean and "Надеюсь подойдёт" in _clean, repr(_clean))
check("an ordinary link is NOT touched",
      GF._strip_fabricated_images("See [the source](https://example.com) for details.")
      == "See [the source](https://example.com) for details.")
check("plain text is unchanged",
      GF._strip_fabricated_images("Просто ответ.") == "Просто ответ.")
# The scrub is an UNCONDITIONAL call in _finalize_answer, so unlike a guarded
# branch there is nothing to short-circuit -- removing it means deleting the
# line, which this catches. (A source check would NOT be enough for a guard:
# putting "False and" in front of one leaves every substring in place.)
check("the scrub is wired into finalisation",
      "_strip_fabricated_images(final_answer)" in _fin_src)

# --- it is wired into the loop --------------------------------------------
src = inspect.getsource(P.personality_node)
check("the loop actually calls the detector", "_detect_silent_failure" in src)

# --- a request the assistant cannot do at all is not «попробуй ещё раз» ----
_unsup = [{"role": "user", "content": "расширь картинку"},
          {"role": "tool", "content": "[TOOL ERROR] Extending the picture (outpaint) is not something this assistant can do"}]
check("an unsupported tool error is recognised", GF._tool_said_unsupported(_unsup))
check("an ordinary tool error is not", not GF._tool_said_unsupported(
    [{"role": "tool", "content": "[TOOL ERROR] ComfyUI server failed"}]))
check("a result from an earlier turn does not count", not GF._tool_said_unsupported(
    _unsup + [{"role": "user", "content": "нарисуй кота"}]))
_line = GF._honest_failure_line({"redraw_image"}, True)
check("the unsupported line does not ask to try again", "ещё раз" not in _line and "не умею" in _line, _line)
check("the ordinary line still does", "ещё раз" in GF._honest_failure_line({"redraw_image"}))

ok = sum(1 for _, o, _ in checks if o)
print(f"\n{ok}/{len(checks)} checks passed")
sys.exit(0 if ok == len(checks) else 1)
