"""Big pictures are sent to the vision model as a bounded copy.

Live, 2026-09-12: a 933x1400 phone photo and a 1920x1088 upscale both made
LM Studio "reload" the model mid-stream and the user got an empty turn.
Pure: no LLM call.

Run: venv/Scripts/python.exe tests/test_vision_sized.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

from PIL import Image
import llm

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


d = tempfile.mkdtemp(prefix="vs_")
small = os.path.join(d, "small.png"); Image.new("RGB", (960, 544), (120, 90, 60)).save(small)
big = os.path.join(d, "big.jpg"); Image.new("RGB", (2000, 3000), (120, 90, 60)).save(big, quality=90)
wide = os.path.join(d, "wide.png"); Image.new("RGBA", (1920, 1088), (120, 90, 60, 200)).save(wide)

check("a render-sized picture is sent as is", llm._vision_sized(small) == small)
out = llm._vision_sized(big)
check("a phone photo is bounded", out != big and max(Image.open(out).size) == llm.VISION_MAX_SIDE, Image.open(out).size)
check("aspect is kept", abs(Image.open(out).size[0] / Image.open(out).size[1] - 2000 / 3000) < 0.01)
check("and it is a JPEG", out.lower().endswith(".jpg"))
out2 = llm._vision_sized(wide)
check("alpha stays PNG", out2.lower().endswith(".png") and Image.open(out2).mode == "RGBA", out2)
check("the copy is reused", llm._vision_sized(big) == out)
check("a missing file falls through to the original", llm._vision_sized(os.path.join(d, "nope.png")).endswith("nope.png"))

# The chat-turn photo travels as BYTES (graph.vision_agent_node's 1536-px
# proxy) and skipped the bound: a 933x1400 upload crashed the model 4/4
# times live on 2026-09-13. Bytes get the same cap as paths.
import io as _io
buf = _io.BytesIO(); Image.new("RGB", (2000, 3000), (120, 90, 60)).save(buf, "JPEG", quality=90)
out_b, mime = llm._vision_sized_bytes(buf.getvalue())
check("big bytes are bounded too", max(Image.open(_io.BytesIO(out_b)).size) == llm.VISION_MAX_SIDE and mime == "image/jpeg",
      Image.open(_io.BytesIO(out_b)).size)
buf = _io.BytesIO(); Image.new("RGB", (960, 544), (1, 2, 3)).save(buf, "PNG")
out_b, mime = llm._vision_sized_bytes(buf.getvalue())
check("small PNG bytes pass through with the right mime", out_b == buf.getvalue() and mime == "image/png")
buf = _io.BytesIO(); Image.new("RGB", (900, 500), (1, 2, 3)).save(buf, "WEBP")
out_b, mime = llm._vision_sized_bytes(buf.getvalue())
check("small webp bytes are re-encoded as JPEG", mime == "image/jpeg" and Image.open(_io.BytesIO(out_b)).format == "JPEG")

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
