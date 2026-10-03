"""The start-up vision self-test: a known picture, a known answer.

2026-09-12: the house model's f16 mmproj was broken -- natural pictures came
through as "dark silhouettes" while a synthetic red-circle/green-square probe
still PASSED. So the probe is a crop of our own render (assets/
vision_selftest.jpg), and the judge is anchored to the two answers measured
on the bad and the good projector. Pure: no LLM call.

Run: venv/Scripts/python.exe tests/test_vision_selftest.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import vision_selftest as V

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


check("the probe picture ships with the app", os.path.exists(V.PROBE_IMAGE), V.PROBE_IMAGE)
check("the probe picture is small", os.path.getsize(V.PROBE_IMAGE) < 200_000)
check("the bad projector's answer fails",
      not V.judge("There is no animal in this picture; the image consists of dark "
                  "silhouettes against a warm background with a white book on a brown surface."))
check("the good projector's answer passes", V.judge("Yes, there is a black cat on a green sofa."))
check("a cat of the wrong colour fails", not V.judge("A ginger cat on a green sofa."))
check("a silhouette answer fails even with the words",
      not V.judge("A black silhouette of a cat on a green sofa."))
check("an empty answer fails", not V.judge(""))
check("under F5_TEST_RUN nothing is asked", V.run(types.SimpleNamespace()) is None)
check("and no thread is started", V.run_in_background(types.SimpleNamespace()) is None)

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(V.__file__))), "core/app_runtime.py"), encoding="utf-8").read()
check("build_runtime starts the self-test after the LLM load",
      "vision_selftest.run_in_background(ctx)" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
