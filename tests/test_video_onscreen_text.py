"""On-screen lettering in video: the script's own words, the model's rendering,
our OCR as the judge, a deterministic overlay when the model garbled it.

No public text LoRA exists for H3 (checked 2026-11-02), so the gate is: one
LLM read says whether the clip must show written text and quotes the exact
strings; after the render our PaddleOCR/EasyOCR reads three frames; a clip
that fails the read gets the exact text burned in as a clean title band.
The classifier is an LLM JSON read — the no-regex rule the project runs on.

Run: venv/Scripts/python.exe tests/test_video_onscreen_text.py
"""
RUN_AS_SCRIPT = True
import os, sys, subprocess, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for d in ("", "media", "imaging", "agent", "core"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
os.environ["F5_TEST_RUN"] = "1"

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {str(extra)[:200]}")

import video
import inspect


# ── the classifier: one LLM read, cached, fail-open ──────────────────────────
def _fake_tm(answer):
    def tm(ctx):
        def call(c, system, user, **kw):
            return answer if "VISIBLE WRITTEN TEXT" in system else ""
        return call
    return tm

print("=" * 70)
print("classifier: LLM JSON read of the script")
print("=" * 70)
video._ONSCREEN_CACHE.clear()
video.text_model = _fake_tm('{"needs": true, "words": ["ПРИВЕТ", "МИР"]}')
check("words quoted verbatim from the script",
      video._onscreen_text_words(None, "неоновая вывеска ПРИВЕТ МИР ночью" + " " * 10)
      == ["ПРИВЕТ", "МИР"])
check("second read of the same script is served from the cache",
      len(video._ONSCREEN_CACHE) == 1)

video.text_model = _fake_tm('{"needs": false, "words": []}')
check("a script without lettering demands none",
      video._onscreen_text_words(None, "кот прыгает со шкафа на диван, мурлычет" + " " * 10) == [])

video.text_model = _fake_tm("the model is having a bad day, no JSON here")
check("garbled LLM reply fails open to no gate",
      video._onscreen_text_words(None, "вывеска СКИДКА на витрине магазина" + " " * 10) == [])

def _raising(ctx):
    def call(*a, **k): raise RuntimeError("llm down")
    return call
video.text_model = _raising
check("LLM down fails open (delivery is never blocked)",
      video._onscreen_text_words(None, "баннер РАСПРОДАЖА над входом" + " " * 10) == [])
src = inspect.getsource(video._onscreen_text_words)
check("the classifier is not a regex/word-list",
      "re.match" not in src and "re.search" not in src)

# ── the OCR read of the finished clip ────────────────────────────────────────
print("=" * 70)
print("OCR judge: reads the delivered frames, fuzzy on one glyph")
print("=" * 70)
import ocr_reader

TMP = tempfile.mkdtemp(prefix="onscreen_")
def _clip(name="c.mp4"):
    p = os.path.join(TMP, name)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=0x304050:s=320x180:d=1:r=12",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", p], check=False)
    return p

clip = _clip()
_real_pieces = ocr_reader.read_pieces
ocr_reader.read_pieces = lambda img: ["ПРИВЕТ МИР НА ВЫВЕСКЕ"]
check("all words read -> pass", video._clip_reads_words(clip, ["ПРИВЕТ", "МИР"]))
check("one misread glyph still passes (fuzzy)",
      video._clip_reads_words(clip, ["ПРИВЕТ", "МНР"]))
check("a word that is not there -> fail",
      not video._clip_reads_words(clip, ["ПРИВЕТ", "ПРОДАНО"]))
ocr_reader.read_pieces = lambda img: []
check("OCR saw nothing -> fail (nothing was drawn)",
      not video._clip_reads_words(clip, ["ПРИВЕТ"]))
def _boom(img): raise RuntimeError("ocr broken")
ocr_reader.read_pieces = _boom
check("OCR itself broken -> pass (checker never blocks delivery)",
      video._clip_reads_words(clip, ["ПРИВЕТ"]))
ocr_reader.read_pieces = _real_pieces

# ── the deterministic overlay ────────────────────────────────────────────────
print("=" * 70)
print("overlay: exact text burned in, audio untouched, source kept")
print("=" * 70)
video._ONSCREEN_CACHE.clear()
video._ONSCREEN_CACHE["вывеска ПРИВЕТ МИР" + " " * 380] = ["ПРИВЕТ", "МИР"]
video.text_model = _fake_tm('{"needs": true, "words": ["ПРИВЕТ", "МИР"]}')
ocr_reader.read_pieces = lambda img: []          # the model drew nothing
out = video._ensure_onscreen_text(None, clip, "вывеска ПРИВЕТ МИР" + " " * 380)
check("the failed clip is replaced by the overlaid one",
      out != clip and out.endswith("_text.mp4") and os.path.exists(out), out)
check("the source clip is left untouched", os.path.exists(clip))
check("no re-encode spiral: a re-run on the overlaid clip idempotent",
      video._ensure_onscreen_text(None, out, "вывеска ПРИВЕТ МИР" + " " * 380).endswith("_text.mp4"))

# now the model DID draw it: no overlay, original delivered
ocr_reader.read_pieces = lambda img: [([[0, 0]], "ПРИВЕТ МИР", 0.99)]
clip2 = _clip("c2.mp4")
video._ONSCREEN_CACHE.clear()
video._ONSCREEN_CACHE["вывеска ПРИВЕТ МИР" + " " * 380] = ["ПРИВЕТ", "МИР"]
check("the model's own lettering passes through unchanged",
      video._ensure_onscreen_text(None, clip2, "вывеска ПРИВЕТ МИР" + " " * 380) == clip2)
# a script with no lettering demand: no gate at all
video._ONSCREEN_CACHE.clear()
video.text_model = _fake_tm('{"needs": false, "words": []}')
check("no text demanded -> clip untouched",
      video._ensure_onscreen_text(None, clip2, "просто котик" + " " * 10) == clip2)
ocr_reader.read_pieces = _real_pieces

# ── the hook sits on every successful render ─────────────────────────────────
gsrc = inspect.getsource(video.generate_video)
check("generate_video runs the lettering gate before returning",
      "_ensure_onscreen_text(ctx, final, description," in gsrc)

# ── text the SOURCE frame already shows is not burned in again ────────────────
from PIL import Image as _PI, ImageDraw as _PD
_srcpic = os.path.join(TMP, "ytseed.png")
_im = _PI.new("RGB", (640, 360), (10, 10, 20))
_dr = _PD.Draw(_im)
_dr.text((40, 30), "GOOD TALKS GREAT PEOPLE", fill=(255, 255, 255))
_im.save(_srcpic)
import ocr_reader as _orc
_real_read = _orc.read
def _read_src(path):
    return (["GOOD", "TALKS", "GREAT", "PEOPLE"]
            if path == _srcpic else _real_read(path))
_orc.read = _read_src
_orc.read_pieces = _read_src          # the source guard reads pieces
video._onscreen_text_words = lambda ctx, d: ["GOOD TALKS GREAT PEOPLE"]
_c3 = _clip("c3.mp4")
out3 = video._ensure_onscreen_text(None, _c3, "player screen", source=_srcpic)
check("a player heading already on the source frame is NOT re-burned",
      out3 == _c3, out3)
video._onscreen_text_words = lambda ctx, d: ["ПРИВЕТ МИР"]
out4 = video._ensure_onscreen_text(None, _c3, "player screen", source=_srcpic)
check("genuinely NEW lettering still earns the overlay",
      out4 != _c3 and out4.endswith("_text.mp4"), out4)
_orc.read = _real_read
video._onscreen_text_words = lambda ctx, d: []

print(f"\n{OK}/{OK+BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
