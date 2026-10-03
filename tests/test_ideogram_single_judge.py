"""Live report (2026-08-05): "image generation is broken — we use Ideogram but I
see old-model in the prompt, it gets stuck in an infinite loop, loads a second
model into memory for some reason, and everything freezes and crashes."

Three separate defects, all in the FRESH-generation path under IMAGE_ENGINE=ideogram4:

1. `generate_image_with_refinement` called `build_image_prompt_from_llm`, whose
   system prompt opens "You are a prompt engineer for the old model
   text-to-image model" and returns the old model prose + the old model steps/cfg. Ideogram
   takes NONE of that — it is driven by ideogram.plan_layout's structured caption
   and IDEOGRAM_STEPS/IDEOGRAM_CFG — so the whole 700-token round was discarded
   and plan_layout re-read the same request anyway. That is the "old-model in the
   prompt" the user saw.

2. Double judging. draw_agent already runs a complete judge-and-repair loop
   (render -> read the lettering back on a vision model -> critique -> rearrange
   the boxes -> render again). The outer refinement loop then judged the SAME
   picture on a different rubric and, on disagreement, re-drew it from scratch —
   re-planning the layout and re-running draw_agent's whole loop — up to
   MAX_IMAGE_REFINEMENT_ATTEMPTS times. Worst case 6 Ideogram renders (two 9GB
   UNETs + a 10GB text encoder each) and ~18 vision calls for one request.

3. Those vision calls alternated between two LM Studio models, because
   TEXT_READ_MODEL pinned the lettering read-back to a different model from the
   chat model. That choice was measured against the OLD 9B house model; the house
   model is now the 26B. On a 24GB card already holding Ideogram's weights there
   is no room for a second resident chat model — the "loads a second model and
   everything freezes" part.

No live LM Studio / ComfyUI / GPU: every boundary is intercepted.

Run: venv/Scripts/python.exe tests/test_ideogram_single_judge.py
"""
import os, sys, tempfile, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import config as C
import image as I
import models
# generate_image_with_refinement LIVES in image_generate and calls its own
# module globals; `image` only re-exports them. Patching image.* therefore
# steered nothing, and this suite made a REAL Ideogram render -- minutes of GPU
# in a suite that is supposed to be deterministic and offline. Patch where the
# call actually resolves.
import image_generate as IG

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


_TMP = tempfile.mkdtemp(prefix="ideogram_single_judge_")
def _fake_png(tag):
    p = os.path.join(_TMP, f"{tag}.png")
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return p


def make_ctx():
    return models.Context(models=None, transcription_cache={}, cache_file=None,
                          asr_lock=threading.Lock(), tts_lock=threading.Lock())


print("=" * 70)
print("1. The user's own words reach the renderer (no prose prompt engineer)")
print("=" * 70)


_rendered = []
def _fake_comfy(ctx, prompt, **kw):
    _rendered.append(prompt)
    # Stand in for the real function's own bookkeeping: a render that went
    # through draw_agent's judged loop reports it here.
    I._ENGINE_JUDGED.update({"judged": True, "score": 9, "ok": True, "problems": []})
    return _fake_png(f"r{len(_rendered)}")

_real_comfy = IG.generate_image_with_comfy
_real_eval = IG.evaluate_image
_eval_calls = []
IG.generate_image_with_comfy = _fake_comfy
IG.evaluate_image = lambda *a, **k: (_eval_calls.append(1) or
                                    {"verdict": "refine", "score": 3, "reason": "x",
                                     "prompt_patch": "more cats"})

_real_engine = C.IMAGE_ENGINE
try:
    C.IMAGE_ENGINE = "ideogram4"
    I._ENGINE_JUDGED.update({"judged": False, "score": 0, "ok": False, "problems": []})
    ctx = make_ctx()
    out = IG.generate_image_with_refinement(ctx, "нарисуй командную строку")

    check("the user's own words went straight to the renderer (plan_layout's input)",
          _rendered and _rendered[0] == "нарисуй командную строку", _rendered)

    print()
    print("=" * 70)
    print("2. A picture the engine already judged is NOT judged a second time")
    print("=" * 70)

    check("evaluate_image was not called at all", not _eval_calls, _eval_calls)
    check("exactly ONE render, not up to 6", len(_rendered) == 1, _rendered)
    check("it reports success", out.get("status") == "success", out)
    check("it carries the ENGINE's own score, not 0/10", out.get("score") == 9, out)
    check("attempts == 1", out.get("attempts") == 1, out)

    print()
    print("=" * 70)
    print("3. An engine loop that ran OUT OF ROUNDS reports partial, honestly")
    print("=" * 70)

    _rendered.clear(); _eval_calls.clear()
    def _fake_comfy_unhappy(ctx, prompt, **kw):
        _rendered.append(prompt)
        I._ENGINE_JUDGED.update({"judged": True, "score": 5, "ok": False,
                                 "problems": ["the sign reads AVCHKE"]})
        return _fake_png(f"u{len(_rendered)}")
    IG.generate_image_with_comfy = _fake_comfy_unhappy
    out2 = IG.generate_image_with_refinement(make_ctx(), "a police car")
    check("status is partial when the engine ran out of repair rounds",
          out2.get("status") == "partial", out2)
    check("the engine's problems are reported, not swallowed",
          "AVCHKE" in (out2.get("reason") or ""), out2)
    check("still only ONE render — no second-rubric re-draw",
          len(_rendered) == 1, _rendered)

    print()
    print("=" * 70)
    print("4. A render that did NOT go through the judged loop is still evaluated")
    print("=" * 70)

    _rendered.clear(); _eval_calls.clear()
    def _fake_comfy_unjudged(ctx, prompt, **kw):
        _rendered.append(prompt)
        I._ENGINE_JUDGED.update({"judged": False, "score": 0, "ok": False, "problems": []})
        return _fake_png(f"p{len(_rendered)}")
    IG.generate_image_with_comfy = _fake_comfy_unjudged
    IG.evaluate_image = lambda *a, **k: (_eval_calls.append(1) or
                                        {"verdict": "success", "score": 9, "reason": ""})
    out3 = IG.generate_image_with_refinement(make_ctx(), "a plain landscape")
    check("the eval pass still runs when the engine did not judge",
          len(_eval_calls) == 1, _eval_calls)
    check("...and its verdict is what gets reported", out3.get("score") == 9, out3)

finally:
    C.IMAGE_ENGINE = _real_engine
    IG.generate_image_with_comfy = _real_comfy
    IG.evaluate_image = _real_eval

print()
print("=" * 70)
print("6. A stale 'already judged' verdict cannot leak into the next render")
print("=" * 70)

import inspect
src = inspect.getsource(I.generate_image_with_comfy)
check("generate_image_with_comfy resets _ENGINE_JUDGED before every render",
      "_ENGINE_JUDGED.update" in src and src.index("_ENGINE_JUDGED.update") < src.index("empty positive prompt"),
      "reset missing or placed after the early return")

print()
print("=" * 70)
print("7. The lettering read-back no longer forces a SECOND resident LLM")
print("=" * 70)

check("TEXT_READ_MODEL defaults to the loaded chat model (empty)",
      C.TEXT_READ_MODEL == "", repr(C.TEXT_READ_MODEL))

import draw_agent
_c = make_ctx()
_c.model_name = "google/gemma-4-26b-a4b-qat"
check("draw_agent._reader hands back the SAME ctx — no model swap per read",
      draw_agent._reader(_c) is _c)

_prev = C.TEXT_READ_MODEL
try:
    C.TEXT_READ_MODEL = "some/other-model"
    check("...but naming one explicitly still switches the reader (still supported)",
          draw_agent._reader(_c) is not _c)
finally:
    C.TEXT_READ_MODEL = _prev

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
