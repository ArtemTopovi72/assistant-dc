"""H3 Context-IR prompt rewrite: accepted only with the required fields, fields
separated by blank lines, on by default in the app and off inside suites."""
from types import SimpleNamespace as NS
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"

import llm
import video as V

ok = []


def check(c, m):
    ok.append(bool(c)); print(("ok   " if c else "FAIL ") + m)


os.environ.pop("VIDEO_CONTEXT_IR", None)
check(V._context_ir_on() is False, "off inside suites by default")
os.environ["VIDEO_CONTEXT_IR"] = "1"
check(V._context_ir_on() is True, "explicit 1 turns it on")
os.environ.pop("F5_TEST_RUN")
os.environ.pop("VIDEO_CONTEXT_IR")
check(V._context_ir_on() is True, "on by default in the app")
os.environ["F5_TEST_RUN"] = "1"

good = ("integrated_multimodal_description: [Shot 1] Live-action, a man...\n"
        "overall_soundscape: birds.\nnon_diegetic_music: N/A")
llm.call_llm_simple = lambda *a, **k: good
out = V.to_context_ir(None, "a man", mode="t2va", seconds=5.17)
check(out.startswith("integrated_multimodal_description:"), "a well-formed rewrite is used")
check("\n\noverall_soundscape:" in out and "\n\nnon_diegetic_music:" in out, "fields separated by a blank line")

llm.call_llm_simple = lambda *a, **k: "Sure! Here is a great video idea about a man."
check(V.to_context_ir(None, "a man", mode="t2va", seconds=5) == "a man", "chatter is rejected, plain prompt kept")
llm.call_llm_simple = lambda *a, **k: None
check(V.to_context_ir(None, "a man", mode="t2va", seconds=5) == "a man", "no answer -> plain prompt kept")

ref_ok = ("subject_definitions:\n<Subject 1> is ...\nsummary:\nx\nretention_analysis:\ny\n"
          "detailed_description:\n[Shot 1] ...\noverall_soundscape: z\nnon_diegetic_music: N/A")
seen = {}
look = NS(vision_model=lambda ctx, image_paths=None, user_text="", system_prompt="", **k:
         seen.update(img=[p for _, p in image_paths or []], sys=system_prompt[:60], user=user_text) or ref_ok)
out = V.to_context_ir(look, "<Picture 1> walks", mode="ref2va", seconds=5, images=["p.png", "guy.png"])
check(out.startswith("subject_definitions:"), "Ref2VA six-section rewrite accepted")
# Shown only the first, the writer guessed the newcomer of <Picture 2> («a dark glove» for
# a light-blue one, only his arm in the shot; 2026-10-12).
check(seen.get("img") == ["p.png", "guy.png"], "the rewriter SEES every reference picture")
check("Full-Reference" in seen.get("sys", ""), "Ref2VA uses the full-reference guide")
# A plan part naming only the newcomer («a man lunges out of the cabinet») left the
# grandma of the first frame out, and H3 rendered the room without her (2026-10-12).
check("every person and animal visible in <Picture 1> is in the shot" in seen.get("user", ""),
      "whoever is in the first frame stays in the shot")
check(V.to_context_ir(NS(vision_model=lambda *a, **k: good), "x", mode="ref2va", seconds=5, images=["p.png"]) == "x",
      "a base-mode answer is not accepted for Ref2VA")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
