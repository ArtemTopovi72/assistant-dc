"""Lettering must come out spelled the way it was asked for.

A police car drawn on 2026-07-29 came back with "AVCHKE" painted on its door
where "POLICE" was asked for, and a second render read "IWABG". The picture was
otherwise fine, which is exactly the problem: the scene critic sees a car with
lettering on it — the thing the layout asked for — and calls it correct. Nobody
was comparing the letters.

Two mechanisms, tested here:

  * the failure is GEOMETRIC first. Six characters in a box 0.09 wide and 0.05
    high is a 1.8:1 box for a string that needs about 3.6:1, at half the height
    a glyph needs. `text_report` / `auto_fix_text` size a lettering box from its
    string before anything is drawn.
  * and it is VERIFIABLE after. `read_text` transcribes what is actually
    painted, `verify_text` compares it to what was requested, and `repair_text`
    redraws with a bigger box and the spelling restated in the caption. That
    comparison OVERRIDES the critic, which is happy either way.

Run: venv/Scripts/python.exe tests/test_ideogram_text_repair.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import inspect
import re
import json

import draw_agent as D
# verify_text and _crop_for_read moved to draw_text.py. verify_text calls
# _crop_for_read through THAT module's globals, so patching
# draw_agent._crop_for_read no longer reaches it -- the stub would be
# ignored, the real crop would run, and this check would go quietly wrong.
# Patch the module that owns the function.
import draw_text as DT
import ideogram as G
import image as I           # imported BEFORE llm is stubbed — image imports from it
import config as C

OK = BAD = 0
def check(name, cond, detail=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {detail}")


def layout(text="POLICE", x=0.30, y=0.55, w=0.09, h=0.05, extra=None):
    """The real failing arrangement: a police car with a lettering box too small
    and the wrong shape for the word it has to carry."""
    els = [{"desc": "a police car seen from the side", "text": "",
            "x": 0.05, "y": 0.35, "w": 0.9, "h": 0.45},
           {"desc": "the door of the car", "text": text, "x": x, "y": y, "w": w, "h": h}]
    if extra:
        els.extend(extra)
    return G.normalize_layout({"background": "a city street", "medium": "photography",
                               "elements": els})


print("=" * 72)
print("THE SHAPE A STRING NEEDS")
print("=" * 72)

w, h = D.text_geometry("POLICE", 0.05)
check("a squashed box is raised to the minimum glyph height", h >= D.MIN_TEXT_H, f"{h}")
check("and its width follows the character count",
      abs(w - h * D.CHAR_ASPECT * 6) < 1e-6, f"{w:.3f}")
check("one character gets a narrow box, not a wide one",
      D.text_geometry("A")[0] < D.text_geometry("ABCDEFGH")[0])
lw, lh = D.text_geometry("A VERY LONG SIGN INDEED OVER HERE", 0.2)
check("a string too long to fit lowers the height instead of crushing the letters",
      lw <= 0.95 and lh < 0.2, f"{lw:.2f}x{lh:.3f}")
check("text_geometry is a pure function of the string and height",
      D.text_geometry("POLICE", 0.05) == D.text_geometry("POLICE", 0.05))

kinds = {p["kind"] for p in D.text_report(layout())}
check("the real failing box is reported as too small", "text_too_small" in kinds, str(kinds))
check("and as the wrong shape for its string", "text_wrong_shape" in kinds, str(kinds))
check("a sentence-length string is reported as too long",
      "text_too_long" in {p["kind"] for p in
                          D.text_report(layout(text="THE QUICK BROWN FOX JUMPED OVER IT"))})
check("an element with no lettering is never a text problem",
      D.text_report(G.normalize_layout(
          {"background": "a field", "elements": [{"desc": "a cow", "x": .1, "y": .1,
                                                  "w": .3, "h": .3}]})) == [])
check("a well-shaped box reports nothing",
      D.text_report(layout(w=0.36, h=0.10)) == [], str(D.text_report(layout(w=0.36, h=0.10))))

fixed, notes = D.auto_fix_text(layout())
el = fixed["elements"][1]
check("auto_fix_text resizes the lettering box", el["h"] >= D.MIN_TEXT_H and el["w"] > 0.25,
      f'{el["w"]:.2f}x{el["h"]:.2f}')
check("it says what it did", any("POLICE" in n for n in notes), str(notes))
check("the repaired box now passes its own report", D.text_report(fixed) == [],
      str(D.text_report(fixed)))
again, notes2 = D.auto_fix_text(fixed)
check("auto_fix_text is idempotent — a second pass changes nothing",
      again["elements"][1] == el and notes2 == [], str(notes2))
check("it keeps the box centred where it was",
      abs((el["x"] + el["w"] / 2) - 0.345) < 0.06, f'{el["x"] + el["w"] / 2:.3f}')
check("it does not touch elements without lettering",
      fixed["elements"][0] == layout()["elements"][0])
check("a box that would overflow the frame is clamped inside it",
      all(e["x"] >= 0 and e["y"] >= 0 and e["x"] + e["w"] <= 1.0001
          and e["y"] + e["h"] <= 1.0001
          for e in D.auto_fix_text(layout(text="EMERGENCY SERVICES", x=0.85, y=0.9))[0]["elements"]))


print()
print("=" * 72)
print("TRANSCRIBE, DON'T READ")
print("=" * 72)

check("case and punctuation are not defects", D._similarity("POLICE", "police.") > 0.99)
check("a garbled render scores low", D._similarity("POLICE", "AVCHKE") < D.TEXT_MATCH_OK,
      f'{D._similarity("POLICE", "AVCHKE"):.2f}')
check("so does the second one we saw", D._similarity("POLICE", "IWABG") < D.TEXT_MATCH_OK,
      f'{D._similarity("POLICE", "IWABG"):.2f}')
check("one wrong letter is still the word", D._similarity("POLICE", "POLIGE") >= D.TEXT_MATCH_OK,
      f'{D._similarity("POLICE", "POLIGE"):.2f}')
check("nothing read at all scores zero", D._similarity("POLICE", "") == 0.0)
check("cyrillic survives normalisation", D._similarity("ПОЛИЦИЯ", "полиция") > 0.99)

src = inspect.getsource(D)
check("the transcription prompt forbids correcting the spelling",
      "Never correct a" in D._TEXT_READ_PROMPT)
# The first version of this prompt illustrated the rule with a concrete fake
# string. The 12B model then transcribed a clean "STOP" as that string: it copied
# the example out of the instructions. A prompt that asks for verbatim output must
# contain nothing quotable.
check("the prompt contains no example string for the model to copy",
      not re.search(r'"[A-Z]{3,}"', D._TEXT_READ_PROMPT), D._TEXT_READ_PROMPT)
check("and it says the instructions are not part of the picture",
      "Copy ONLY from the picture" in D._TEXT_READ_PROMPT)
check("the vision call passes no prefill argument (it breaks vision calls)",
      "prefill=" not in inspect.getsource(D.read_text))


class _LLM(types.ModuleType):
    """Stand-in for the llm module: returns whatever the test queued."""
    def __init__(self, reply):
        super().__init__("llm")
        self.reply = reply
        self.calls = []
    def analyze_image_with_llm(self, ctx, **kw):
        self.calls.append(dict(kw, ctx_model=getattr(ctx, "model_name", None)))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply
    def send_to_lm_studio(self, ctx, messages, **kw):
        self.calls.append(dict(kw, messages=messages))
        if isinstance(self.reply, Exception):
            raise self.reply
        return {"content": self.reply}

def with_llm(reply):
    sys.modules["llm"] = _LLM(reply)
    return sys.modules["llm"]

_saved_llm = sys.modules.get("llm")
CTX = types.SimpleNamespace(is_cancelled=lambda: False, model_name="chat/model")
# A real dataclass Context for the proxy tests — a namespace would hide the
# attribute-forwarding the proxy exists to do.
import threading
from pathlib import Path
from models import Context
CTX_REAL = Context(models=None, transcription_cache={}, cache_file=Path("tests/_x.json"),
                   asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                   model_name="chat/model")
try:
    with_llm('{"strings": ["AVCHKE", "911"]}')
    check("read_text returns the strings the model transcribed",
          D.read_text(CTX, "x.png") == ["AVCHKE", "911"])
    with_llm('here you go:\n```json\n{"strings":["OPEN"]}\n```')
    check("a fenced, chatty reply is still parsed", D.read_text(CTX, "x.png") == ["OPEN"])
    with_llm('{"strings": []}')
    check("no lettering is an empty list, not None", D.read_text(CTX, "x.png") == [])
    with_llm('the sign says POLICE')
    check("an unparseable reply is 'could not look', not 'no text'",
          D.read_text(CTX, "x.png") is None)
    with_llm('{"answer": "POLICE"}')
    check("the wrong JSON shape is also 'could not look'",
          D.read_text(CTX, "x.png") is None)
    with_llm(RuntimeError("vision is down"))
    check("a vision exception never escapes", D.read_text(CTX, "x.png") is None)
    check("no image means no call", D.read_text(CTX, "") is None)
    check("no model means no call", D.read_text(None, "x.png") is None)

    # The transcription must NOT run on the chat model. Measured live on eight
    # frames: the 9B chat model read a clean "STOP" as "HIT" and scored garbled
    # renders above correct ones (correct min 0.29 vs garbled max 0.36 — the two
    # distributions overlapped, so the check would have redrawn good pictures at
    # random). gemma-4-12b read all eight exactly: correct 1.00, garbled <= 0.50.
    _rounds_model = C.TEXT_READ_MODEL
    try:
        C.TEXT_READ_MODEL = "reader/model"
        seen = {}
        m = with_llm('{"strings": ["OPEN"]}')
        m.analyze_image_with_llm = lambda ctx, **kw: (
            seen.update(model=ctx.model_name) or '{"strings": ["OPEN"]}')
        D.read_text(CTX_REAL, "x.png")
        check("the transcription runs on the dedicated reader model",
              seen.get("model") == "reader/model", str(seen))
        check("the real context keeps its own model", CTX_REAL.model_name != "reader/model")
        proxy = D._reader(CTX_REAL)
        proxy.last_api_call_time = 12345.0
        check("writes through the proxy land on the real context, not a copy",
              CTX_REAL.last_api_call_time == 12345.0)
        check("and every other attribute is the real one",
              proxy.api_lock is CTX_REAL.api_lock)
        C.TEXT_READ_MODEL = ""
        check("an empty setting uses the chat model, no proxy",
              D._reader(CTX_REAL) is CTX_REAL)
        C.TEXT_READ_MODEL = CTX_REAL.model_name
        check("and so does naming the model that is already loaded",
              D._reader(CTX_REAL) is CTX_REAL)
    finally:
        C.TEXT_READ_MODEL = _rounds_model

    print()
    print("=" * 72)
    print("COMPARING WHAT WAS ASKED FOR WITH WHAT WAS PRINTED")
    print("=" * 72)

    with_llm('{"strings": ["AVCHKE"]}')
    v = D.verify_text(CTX, "x.png", layout())
    check("the garbled render is caught", v["ok"] is False and v["failures"] == [1], str(v))
    check("the evidence is kept, both sides of it",
          v["checks"][0]["expected"] == "POLICE" and v["checks"][0]["read"] == "AVCHKE")
    check("and turned into a sentence a user can read",
          "“POLICE”" in D.text_problems(v["checks"])[0]
          and "“AVCHKE”" in D.text_problems(v["checks"])[0],
          str(D.text_problems(v["checks"])))

    with_llm('{"strings": ["Police"]}')
    check("a correct render passes", D.verify_text(CTX, "x.png", layout())["ok"] is True)
    with_llm('{"strings": []}')
    v = D.verify_text(CTX, "x.png", layout())
    check("lettering that never appeared is a failure too", v["ok"] is False)
    check("and it says so without inventing a reading",
          "no readable lettering" in D.text_problems(v["checks"])[0])

    with_llm('{"strings": ["POLICE"]}')
    two = layout(extra=[{"desc": "a sign", "text": "POLICE", "x": 0.1, "y": 0.05,
                         "w": 0.3, "h": 0.1}])
    v = D.verify_text(CTX, "x.png", two)
    check("one rendered string cannot satisfy two elements asking for it",
          v["ok"] is False and len(v["failures"]) == 1, str(v["failures"]))

    with_llm('nonsense')
    v = D.verify_text(CTX, "x.png", layout())
    check("a read-back that could not run does not condemn the picture",
          v["ok"] is True and v["source"] == "unavailable", str(v))
    with_llm('{"strings": ["ANYTHING"]}')
    v = D.verify_text(CTX, "x.png", G.normalize_layout(
        {"background": "a field", "elements": [{"desc": "a cow", "x": .1, "y": .1,
                                                "w": .3, "h": .3}]}))
    check("a picture with no lettering skips the vision call entirely",
          v["source"] == "no_text" and sys.modules["llm"].calls == [], str(v))

    # Reading the WHOLE picture does not survive a real render: on a 1024x1024
    # street scene the reader enumerated "POLICE (partially visible)" forty times
    # and hit the token limit before emitting any JSON — reported as "could not
    # look", so the check silently switched itself off. Each element is now read
    # from its own region, which the same model answers in one line.
    import tempfile
    from PIL import Image
    big = os.path.join(tempfile.mkdtemp(prefix="textread_"), "render.png")
    Image.new("RGB", (1024, 1024), (200, 200, 205)).save(big)

    m = with_llm('{"strings": ["POLICE"]}')
    v = D.verify_text(CTX, big, layout())
    # Two region reads now: the element's own box, and its host's box for
    # lettering nobody asked for. Neither is the whole frame.
    check("the lettering is read from the element's own region, not the whole frame",
          v["ok"] is True and 1 <= len(m.calls) <= 2
          and all(c["image_path"] != big for c in m.calls),
          [c.get("image_path") for c in m.calls])
    crop_path = m.calls[0]["image_path"]
    check("and the crop is cleaned up afterwards", not os.path.exists(crop_path), crop_path)

    el = layout()["elements"][1]
    c = D._crop_for_read(big, el)
    try:
        check("the crop covers the box with room around it, inside the frame",
              c and Image.open(c).size[0] > el["w"] * 1024, str(Image.open(c).size))
        check("a small region is enlarged so the glyphs survive the encoder",
              max(Image.open(c).size) >= 512, str(Image.open(c).size))
    finally:
        if c: os.unlink(c)
    check("an unreadable render yields no crop rather than raising",
          D._crop_for_read(os.path.join(os.path.dirname(big), "gone.png"), el) is None)
    check("a degenerate box yields no crop",
          D._crop_for_read(big, {"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0}) is None)

    # ...and when the crop cannot be read, the whole picture is still tried once.
    m = with_llm('{"strings": ["POLICE"]}')
    _real_crop = DT._crop_for_read
    try:
        DT._crop_for_read = lambda p, b: None
        v = D.verify_text(CTX, big, layout(extra=[{"desc": "a sign", "text": "STOP",
                                                   "x": .1, "y": .05, "w": .3, "h": .1}]))
        check("a failed crop falls back to reading the whole picture, once for all "
              "elements", len(m.calls) == 1 and m.calls[0]["image_path"] == big,
              f"{len(m.calls)} call(s)")
    finally:
        DT._crop_for_read = _real_crop

    print()
    print("=" * 72)
    print("THE LAYOUT-ONLY PARSER WAS EATING EVERY NON-LAYOUT ANSWER")
    print("=" * 72)

    # ideogram._extract_json only accepts objects carrying LAYOUT keys. Three
    # callers pass it something else entirely, so a perfectly good answer scored
    # zero and was discarded. Measured live on 2026-07-29: the critic ended its
    # reply with {"ok": true, "score": 10, ...} and still reported "unavailable" —
    # it had never once judged a real picture.
    import numpy as np
    noisy = os.path.join(os.path.dirname(big), "noisy.png")
    Image.fromarray(np.random.default_rng(3).integers(
        0, 255, (512, 512, 3), dtype=np.uint8)).save(noisy)

    m = with_llm('<|channel>thought\nLet me check each element.\n'
                 '<channel|>{"ok": false, "score": 3, "problems": ["the dog is missing"], '
                 '"ops": [{"op": "add", "desc": "a dog"}]}')
    v = D.critique(CTX, noisy, layout())
    check("a verdict with no layout keys is no longer thrown away",
          v["source"] == "vision" and v["ok"] is False, str(v))
    check("its problems survive", v["problems"] == ["the dog is missing"], str(v["problems"]))
    check("and so do its repair ops", len(v["ops"]) == 1, str(v["ops"]))
    check("the critic also runs on the reader model",
          m.calls[-1]["ctx_model"] == C.TEXT_READ_MODEL or not C.TEXT_READ_MODEL,
          str(m.calls[-1].get("ctx_model")))

    m = with_llm('{"ops": [{"op": "delete", "target": "the door of the car"}]}')
    _, notes = D.edit_layout(CTX, layout(), "remove the door")
    check("an edit answer is no longer discarded as 'no change proposed'",
          any("deleted" in n for n in notes), str(notes))

    m = with_llm('{"region": "the old man", "content": "an elderly woman", "removal": false}')
    spec = D.split_edit(CTX, "swap the grandpa for a grandma")
    check("a contained-edit split is no longer discarded",
          spec["region"] == "the old man" and spec["content"] == "an elderly woman",
          str(spec))

    print()
    print("=" * 72)
    print("THE REPAIR")
    print("=" * 72)

    rep, notes = D.repair_text(layout(), [1], 0)
    el = rep["elements"][1]
    check("the box grows past what the string minimally needs",
          el["h"] > D.MIN_TEXT_H and el["w"] > 0.3, f'{el["w"]:.2f}x{el["h"]:.2f}')
    check("the description names the word and its letter count (no hyphens: they get painted)",
          "6-letter word “POLICE”" in el["desc"] and "P-O" not in el["desc"], el["desc"])
    check("and still says what the surface is",
          "door" in el["desc"].lower(), el["desc"])
    check("the string itself is untouched", el["text"] == "POLICE")
    check("the repair explains itself", any("garbled" in n for n in notes), str(notes))
    rep2, _ = D.repair_text(layout(), [1], 1)
    check("a second failure buys more room than the first",
          rep2["elements"][1]["h"] > el["h"],
          f'{el["h"]:.3f} -> {rep2["elements"][1]["h"]:.3f}')
    check("repairing an element that does not exist is a no-op, not a crash",
          D.repair_text(layout(), [99], 0)[1] == [])
    check("repairing a non-text element is skipped",
          D.repair_text(layout(), [0], 0)[1] == [])
    twice, _ = D.repair_text(rep, [1], 0)
    check("re-spelling an already-spelled description does not nest the parenthetical",
          twice["elements"][1]["desc"].count("letter word") == 1,
          twice["elements"][1]["desc"])

    print()
    print("=" * 72)
    print("THE LOOP: DRAW, READ IT BACK, REDRAW")
    print("=" * 72)

    renders = []
    reads = ["AVCHKE", "POLICE"]

    def fake_generate(ctx, prompt, **kw):
        renders.append(G.caption_to_layout(kw["caption"]))
        return f"/tmp/render{len(renders)}.png"

    _real_gen, _real_crit = G.generate, D.critique
    try:
        G.generate = fake_generate
        # The critic is HAPPY every time — that is the real bug. Only the
        # transcription can tell the two renders apart.
        D.critique = lambda ctx, img, lay, evidence=None: {"ok": True, "score": 9, "problems": [],
                                            "ops": [], "source": "vision"}
        sys.modules["llm"] = _LLM(None)
        sys.modules["llm"].analyze_image_with_llm = lambda ctx, **kw: json.dumps(
            {"strings": [reads[min(len(renders), len(reads)) - 1]]})
        res = D.run(CTX, "a police car", layout=layout(), rounds=1)
    finally:
        G.generate, D.critique = _real_gen, _real_crit

    check("a garbled first render is redrawn even though the critic approved it",
          len(renders) == 2, f"{len(renders)} render(s)")
    check("the second attempt is given a bigger lettering box",
          renders[1]["elements"][1]["h"] > renders[0]["elements"][1]["h"],
          f'{renders[0]["elements"][1]["h"]:.3f} -> {renders[1]["elements"][1]["h"]:.3f}')
    check("and a caption that spells the word out",
          "6-letter word" in renders[1]["elements"][1]["desc"])
    check("the loop returns the corrected render", res["image"] == "/tmp/render2.png")
    check("it stops once the words come out right", res["stopped"] == "ok", str(res["stopped"]))
    check("the read-back verdict is reported alongside the picture",
          res["text"]["ok"] is True and res["text"]["source"] == "vision", str(res.get("text")))
    check("the history keeps the failed attempt and why it failed",
          len(res["history"]) == 2
          and any("AVCHKE" in p for p in res["history"][0]["verdict"]["problems"]),
          str(res["history"][0]["verdict"].get("problems")))

    # ...and the layout that was never garbled must not be redrawn for nothing.
    renders.clear()
    try:
        G.generate = fake_generate
        D.critique = lambda ctx, img, lay, evidence=None: {"ok": True, "score": 9, "problems": [],
                                            "ops": [], "source": "vision"}
        with_llm('{"strings": ["POLICE"]}')
        res = D.run(CTX, "a police car", layout=layout(), rounds=2)
    finally:
        G.generate, D.critique = _real_gen, _real_crit
    check("a correct render is not redrawn", len(renders) == 1, f"{len(renders)}")
    check("the first render already had a properly shaped box (fixed before drawing)",
          D.text_report(renders[0]) == [], str(D.text_report(renders[0])))

    # The critic gets a vote on the arrangement, and its ops are applied AFTER the
    # lettering repair — so a critic that "helpfully" shrinks the sign would hand
    # the next attempt the very geometry that garbled the word in the first place.
    renders.clear()
    try:
        G.generate = fake_generate
        D.critique = lambda ctx, img, lay, evidence=None: {
            "ok": False, "score": 4, "problems": ["the lettering is squeezed"],
            "ops": [{"op": "resize", "target": "the door of the car",
                     "w": 0.06, "h": 0.04}], "source": "vision"}
        with_llm('{"strings": ["AVCHKE"]}')
        D.run(CTX, "a police car", layout=layout(), rounds=1)
    finally:
        G.generate, D.critique = _real_gen, _real_crit
    check("a critic op cannot shrink a lettering box back below what its string needs",
          len(renders) == 2 and D.text_report(renders[1]) == [],
          str(D.text_report(renders[1]) if len(renders) > 1 else renders))

    # A vision outage must not turn every lettered picture into three renders.
    renders.clear()
    try:
        G.generate = fake_generate
        D.critique = lambda ctx, img, lay, evidence=None: {"ok": True, "score": 9, "problems": [],
                                            "ops": [], "source": "vision"}
        with_llm(RuntimeError("vision is down"))
        res = D.run(CTX, "a police car", layout=layout(), rounds=2)
    finally:
        G.generate, D.critique = _real_gen, _real_crit
    check("a read-back outage costs one render, not the whole budget",
          len(renders) == 1 and res["text"]["source"] == "unavailable", f"{len(renders)}")


    print()
    print("=" * 72)
    print("ROUTING: THE AGENT'S OWN DRAW PATH GOES THROUGH THE LOOP")
    print("=" * 72)

    calls = {"run": 0, "gen": 0, "caption": None}

    def fake_run(ctx, request="", **kw):
        calls["run"] += 1
        calls["layout"] = kw.get("layout")
        return {"image": "/tmp/loop.png", "layout": kw.get("layout"), "history": [],
                "problems": [], "stopped": "ok"}

    def fake_gen2(ctx, prompt, **kw):
        calls["gen"] += 1
        calls["caption"] = kw.get("caption")
        return "/tmp/single.png"

    _real_plan, _rounds, _real_run = G.plan_layout, C.IDEOGRAM_TEXT_ROUNDS, D.run
    try:
        D.run, G.generate = fake_run, fake_gen2
        C.IDEOGRAM_TEXT_ROUNDS = 1

        G.plan_layout = lambda ctx, p: layout()
        out = I._ideogram_draw(CTX, "a police car", width=1024, height=1024,
                               seed=1, timeout=60)
        check("a scene with lettering is drawn through the read-back loop",
              out == "/tmp/loop.png" and calls["run"] == 1 and calls["gen"] == 0,
              str(calls))

        calls.update(run=0, gen=0)
        G.plan_layout = lambda ctx, p: G.normalize_layout(
            {"background": "a field", "elements": [{"desc": "a cow", "x": .1, "y": .1,
                                                    "w": .3, "h": .3}]})
        out = I._ideogram_draw(CTX, "a cow", width=1024, height=1024, seed=1, timeout=60)
        check("a scene with no lettering is a single submit, as before",
              out == "/tmp/single.png" and calls["run"] == 0 and calls["gen"] == 1,
              str(calls))
        check("and it reuses the layout it just planned instead of planning twice",
              isinstance(calls["caption"], dict)
              and calls["caption"]["compositional_deconstruction"]["elements"][0]["desc"]
                  == "a cow", str(calls["caption"]))

        calls.update(run=0, gen=0)
        C.IDEOGRAM_TEXT_ROUNDS = 0
        G.plan_layout = lambda ctx, p: layout()
        out = I._ideogram_draw(CTX, "a police car", width=1024, height=1024,
                               seed=1, timeout=60)
        check("the check can be turned off entirely",
              out == "/tmp/single.png" and calls["run"] == 0 and calls["gen"] == 1,
              str(calls))

        calls.update(run=0, gen=0)
        C.IDEOGRAM_TEXT_ROUNDS = 1
        def boom(ctx, p): raise RuntimeError("planner exploded")
        G.plan_layout = boom
        out = I._ideogram_draw(CTX, "a police car", width=1024, height=1024,
                               seed=1, timeout=60)
        check("a failure anywhere in the new path still returns a picture",
              out == "/tmp/single.png" and calls["gen"] == 1, str(calls))

        calls.update(run=0, gen=0)
        G.plan_layout = lambda ctx, p: layout()
        D.run = lambda ctx, request="", **kw: {"image": None, "stopped": "refused",
                                               "problems": ["declined"], "history": [],
                                               "layout": kw.get("layout")}
        try:
            I._ideogram_draw(CTX, "x", width=64, height=64, seed=1, timeout=60)
            refused = False
        except G.ContentRefused:
            refused = True
        check("a refusal is raised, not silently redrawn on another engine", refused)
    finally:
        G.plan_layout, C.IDEOGRAM_TEXT_ROUNDS = _real_plan, _rounds
        G.generate, D.run = _real_gen, _real_run
finally:
    if _saved_llm is not None:
        sys.modules["llm"] = _saved_llm
    else:
        sys.modules.pop("llm", None)


print()
print("=" * 72)
print("WHAT THE PLANNER IS TOLD ABOUT LETTERING")
print("=" * 72)

p = G._PLANNER_PROMPT
check("the planner is told to put the exact string in a text field",
      "exact string" in p.lower())
check("it is told to keep it short", "SHORT" in p)
check("it is told the desc describes the surface, not the words",
      "never a rewording" in p.lower())
check("it is given the box geometry rule, with the numbers",
      "0.08" in p and "0.6" in p)
check("it is told one string per element", "One string per element" in p)
check("and told not to invent lettering", "Do not invent lettering" in p)

print()
print("=" * 72)
print("THE PLANNER FORGETS — THE STRING IS PUT BACK ANYWAY")
print("=" * 72)

# Live, 2026-07-29: 'a neon sign above the door reading "CAFE ROSA"' was planned
# as six elements with no `text` field anywhere, so nothing downstream knew the
# picture had lettering — no sizing, no read-back, no repair. Routing cannot
# depend on the planner remembering.
check("a quoted string asked for as a sign is picked up",
      G.requested_strings('a neon sign above the door reading "CAFE ROSA"') == ["CAFE ROSA"])
check("so is one asked for as a word", G.requested_strings(
    'a police car with the word "POLICE" on its door') == ["POLICE"])
check("russian quotes and cue words work too",
      G.requested_strings("нарисуй вывеску с надписью «ОТКРЫТО»") == ["ОТКРЫТО"])
check("a quote with no lettering cue is NOT painted into the picture",
      G.requested_strings('draw a "cat" on a sofa') == [])
check("a plain request has no strings", G.requested_strings("a quiet street at dusk") == [])
check("several strings come back in order", G.requested_strings(
    'a shop with a sign reading "OPEN" and a banner saying "SALE"') == ["OPEN", "SALE"])
check("an empty prompt is handled", G.requested_strings("") == [])

_base = G.normalize_layout({"background": "a street at night", "elements": [
    {"desc": "a small corner cafe storefront", "x": .1, "y": .2, "w": .6, "h": .5},
    {"desc": "people at outdoor tables", "x": .1, "y": .6, "w": .8, "h": .3}]})
_req = 'a neon sign above the door reading "CAFE ROSA"'
out = G.ensure_text_elements(_base, _req)
letter = [e for e in out["elements"] if e["text"]]
check("the dropped string becomes a text element", len(letter) == 1
      and letter[0]["text"] == "CAFE ROSA", str(letter))
check("it is placed on the surface the user named",
      "storefront" in letter[0]["desc"], letter[0]["desc"])
check("the surface itself is NOT reshaped into a letterbox",
      out["elements"][0]["w"] == 0.6 and out["elements"][0]["h"] == 0.5,
      str(out["elements"][0]))
check("the new box is already the right shape for its string",
      D.text_report(out) == [], str(D.text_report(out)))
check("and does not create an arrangement problem",
      [p["kind"] for p in D.geometry_report(out)] == [],
      str([p["kind"] for p in D.geometry_report(out)]))
check("running it twice adds nothing", G.ensure_text_elements(out, _req) == out)
check("a string the planner DID place is left alone",
      len([e for e in G.ensure_text_elements(out, _req)["elements"] if e["text"]]) == 1)
noscene = G.ensure_text_elements(G.normalize_layout(
    {"background": "a field", "elements": [{"desc": "a cow", "x": .1, "y": .1,
                                            "w": .3, "h": .3}]}), _req)
check("with no plausible surface the lettering still gets an element of its own",
      any(e["text"] == "CAFE ROSA" for e in noscene["elements"]), str(noscene["elements"]))
check("a prompt asking for no lettering leaves the layout untouched",
      G.ensure_text_elements(_base, "a quiet street at dusk") == _base)

# The backstop has to be on the SUCCESS path, not just the fallback: the live
# failure was a planner that answered perfectly well and simply left `text` out.
_real_ask = G._ask_planner
try:
    G._ask_planner = lambda ctx, prompt, attempts=2: {
        "background": "a street at night", "medium": "photography",
        "aesthetics": "moody", "lighting": "neon",
        "elements": [{"desc": "a corner cafe storefront", "x": .1, "y": .2,
                      "w": .6, "h": .5}]}
    planned = G.plan_layout(object(), _req)
    check("a planner that answers well but forgets the lettering is corrected",
          any(e["text"] == "CAFE ROSA" for e in planned["elements"]),
          str(planned["elements"]))
    G._ask_planner = lambda ctx, prompt, attempts=2: None
    fell_back = G.plan_layout(object(), _req)
    check("and so is the fallback layout",
          any(e["text"] == "CAFE ROSA" for e in fell_back["elements"]),
          str(fell_back["elements"]))
finally:
    G._ask_planner = _real_ask

cap = G.build_caption("a street", [G.element("a sign", G.bbox(.1, .1, .3, .1),
                                             text="OPEN")], art_style="cartoon",
                      medium="illustration")
check("a text element is typed as text",
      cap["compositional_deconstruction"]["elements"][0]["type"] == "text")
check("the caption carries the literal string",
      cap["compositional_deconstruction"]["elements"][0]["text"] == "OPEN")
_real_ask2 = G._ask_planner
G._ask_planner = lambda ctx, prompt, attempts=2: {
    "background": "a person", "art_style": "caricature", "medium": "",
    "elements": [{"desc": "a person", "x": .1, "y": .1, "w": .6, "h": .6}]}
try:
    cap_style = G.plan_caption(object(), "draw a caricature of a person")
finally:
    G._ask_planner = _real_ask2
# A behavior check, not a literal source-text scan: the earlier version of this
# check (`"art_style=" in inspect.getsource(G.plan_caption)`) broke when
# plan_caption was refactored to route through layout_to_caption (which DOES
# still pass art_style through) instead of building the caption by hand —
# the string literal moved to a different function while the behavior stayed
# correct, so the check needs to prove the OUTCOME, not grep the source.
check("plan_caption no longer drops art_style (a caricature was captioned as a photo)",
      cap_style["style_description"]["art_style"] == "caricature",
      cap_style["style_description"])

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
