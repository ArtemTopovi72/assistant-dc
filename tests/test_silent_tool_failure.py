"""Regression: a tool that fails QUIETLY must not become "Готово!".

Found with bench/tc_chaos.py, which breaks tools on purpose and scores only
what the user is finally told. The loop classified a result as failed if and
only if it began with "[TOOL ERROR]", so the polite failures were handled well
(11 of 13 runs reported honestly) and the silent ones were not: against tools
returning "" or a truncated string, the agent said "Готово! Я добавил ей очки"
on 12 of 21 runs. The tool had produced nothing at all.

A model cannot notice an absence. It has to be shown one, which is what
_detect_silent_failure does -- with two signals that are cheap and certain:

  * a blank result is never a real answer;
  * a tool whose job is to produce a file, that finished with no path in state,
    produced no file, whatever its prose claims. Those state keys are the ones
    tg_bot and the GUI actually read to deliver the artifact, so an unset key
    means the user receives nothing.

Deliberately NOT attempted here: plausible-but-wrong output with the artifact
present. That is undetectable by inspection of the result, needs the vision
verifier, and is left as the known remaining gap (chaos mode "wrong" still
scores 3 false successes).

Run: .\venv\Scripts\python.exe tests\test_silent_tool_failure.py
"""
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

# Handing over a result is the same lie as claiming to have made one, and
# _ACTION_CLAIM_RE (first-person verbs only) does not see it.
DELIVERY = [
    ("vot_your", "Вот ваш рыжий кот в скафандре среди звезд!"),
    ("vot_yours", "Вот твой рыжий кот-астронавт!"),
    ("vot_photo", "Вот реальное фото Эйфелевой башни."),
    ("gotovo", "Готово!"),
    ("english", "Here is your image."),
]
for name, text in DELIVERY:
    check(f"delivery_claim_{name}", bool(P._DELIVERY_CLAIM_RE.search(text)),
          repr(text[:50]))

# It must stay narrow: these are honest answers given AFTER a failure, and
# flagging them would push the agent to retry a thing it correctly gave up on.
NOT_DELIVERY = [
    ("deck_plan", "Вот план презентации из 8 слайдов: 1. Основание Рима"),
    ("ideas", "Вот несколько идей для названия кофейни"),
    ("apology", "К сожалению, мне не удалось создать изображение."),
    ("physics", "Небо голубое из-за рэлеевского рассеяния."),
]
for name, text in NOT_DELIVERY:
    check(f"not_delivery_{name}", not P._DELIVERY_CLAIM_RE.search(text),
          repr(text[:50]))

# The decision itself, not a grep of the loop's source: an earlier version of
# this test only checked that the guard's substrings appeared in
# personality_node, and putting "False and" in front of the whole condition
# left it passing. V() is the real predicate the loop calls.
V = lambda draft, neg, sent: P._answer_contradicts_reality(draft, neg, set(), sent)
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

# --- the third phrasing: the deliverable "exists" --------------------------
# Neither a first-person verb nor a "вот …". This is where the lies hid the
# longest, and where the chaos scorer was hiding them too: "Презентация про
# историю Рима на 8 слайдов готова" with no file, and "Результат вычисления …
# равен 20 965 334.4" when calculate had returned nothing (the real value of
# 18432*977+15% is 20 709 273.6, and the invented one differed every run).
PROMISE = [
    ("deck_ready", "Презентация про историю Рима на 8 слайдов готова."),
    ("file_ready", "Файл презентации готов."),
    ("computed", "Результат вычисления 18432 * 977 + 15% равен 20 965 334.4"),
    ("equals", "равен 20963112"),
    ("english", "The file is ready."),
]
for name, text in PROMISE:
    check(f"promise_claim_{name}", bool(P._PROMISE_CLAIM_RE.search(text)),
          repr(text[:55]))

NOT_PROMISE = [
    ("plan_offered", "Вот план презентации из 8 слайдов: 1. Основание Рима"),
    ("apology", "К сожалению, не удалось создать файл."),
    ("explain", "Презентацию можно собрать из восьми разделов."),
]
for name, text in NOT_PROMISE:
    check(f"not_promise_{name}", not P._PROMISE_CLAIM_RE.search(text),
          repr(text[:55]))

# A broken promise is a trigger on its own — no verification needed.
A = P._answer_contradicts_reality
check("a failed artifact tool + a promise claim is caught",
      A("Презентация готова.", False, {"create_presentation"}, False))
check("a failed calculate + an invented number is caught",
      A("Результат вычисления равен 20963112", False, {"calculate"}, False))
check("no failure and no negative verdict lets it through",
      not A("Презентация готова.", False, set(), False))
check("a failed tool alone is not enough without a claim",
      not A("К сожалению, не получилось.", False, {"create_presentation"}, False))

# calculate must return the VALUE, the way tools._handle_calculate does
# (str(result)). Checking merely for a digit was not enough -- chaos mode
# "wrong" hands back a numbered list, which has digits, sailed through, and the
# model then reported "Результат вычисления: 20 962 144.8" against a true value
# of 20 709 273.6. Parsing it as a number is the handler's own contract.
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
# Chosen over another corrective round because the two measured very
# differently: the deterministic scrub of fabricated image links let ZERO
# through in 140 chaos runs, while a corrective round is advice the model is
# free to ignore -- and did, on 13 of 28 route_calc runs.
SR = GF._STATED_RESULT_RE
for label, text in [("vychisleniya", "Результат вычисления 18432 * 977 равен 20963112"),
                    ("colon", "Результат: 21 045 897.6"),
                    ("equals", "18432 * 977 = 18008064")]:
    check(f"stated_result_{label}", bool(SR.search(text)), text[:45])
for label, text in [("cats", "У неё 2 кота и 3 собаки"),
                    ("prose", "Небо голубое из-за рассеяния."),
                    ("price", "Цена равна 500 рублей")]:
    check(f"not_stated_result_{label}", not SR.search(text), text[:45])

# A fragment with no letters ("4.", "0.", ".") is what a small model emits
# after the corrective round pops its draft. Worse for the user than an honest
# failure, so it gets the same replacement.
check("a degenerate fragment has no letters",
      not any(c.isalpha() for c in "4.") and not any(c.isalpha() for c in "0."))
check("the honest replacement refuses to name a number",
      not GF._STATED_RESULT_RE.search(GF._NO_CALC_ANSWER), GF._NO_CALC_ANSWER)
# --- the same backstop for the tools that hand over a FILE -----------------
# The corrective round asks the model not to claim a result that does not
# exist; it claimed one anyway on 9 of 27 lying runs ("Вот реальное фото
# Эйфелевой башни" after find_photo failed). This is the deterministic net
# under it.
check("a failed find_photo produces an honest line",
      "фотографию" in GF._honest_failure_line({"find_photo"}))
check("several failures are named together",
      "презентацию" in GF._honest_failure_line({"create_presentation",
                                                "generate_image"})
      and "картинку" in GF._honest_failure_line({"create_presentation",
                                                 "generate_image"}))
check("a tool that hands over no file gets no line",
      GF._honest_failure_line({"search"}) == "")
check("the honest line makes no claim of its own",
      not P._DELIVERY_CLAIM_RE.search(GF._honest_failure_line({"find_photo"}))
      and not P._PROMISE_CLAIM_RE.search(GF._honest_failure_line({"find_photo"})),
      GF._honest_failure_line({"find_photo"}))
# "не получилось" must survive the negation lookbehind added earlier, or the
# replacement text would itself read as a false success.
check("the honest line is not itself a delivery claim",
      not P._DELIVERY_CLAIM_RE.search("Не получилось сделать фотографию"))
check("every artifact tool has a noun",
      set(GF._PROMISE_NOUN) == set(P._TOOL_ARTIFACT),
      f"{sorted(set(P._TOOL_ARTIFACT) - set(GF._PROMISE_NOUN))} missing")

_fin_src = __import__("inspect").getsource(GF._finalize_answer)
check("the file backstop is wired",
      "_honest_failure_line(_artifact_fails)" in _fin_src
      and "final_answer = _line" in _fin_src)
check("the replacement is wired with both triggers",
      "_STATED_RESULT_RE.search(final_answer) or _degenerate" in _fin_src)
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
      "_strip_fabricated_images(final_answer)"
      in __import__("inspect").getsource(GF._finalize_answer))

# --- it is wired into the loop --------------------------------------------
import inspect
src = inspect.getsource(P.personality_node)
check("the loop actually calls the detector", "_detect_silent_failure" in src)

ok = sum(1 for _, o, _ in checks if o)
print(f"\n{ok}/{len(checks)} checks passed")
sys.exit(0 if ok == len(checks) else 1)
