"""New lettering on a photo: FireRed renders, OCR reads back, a wrong year reseeds.

Night bench 2026-09-24: FireRed painted "УРОЖАЙ 2025" for "УРОЖАЙ 2026" on
four variants in a row. Delivered unread, that is a false success.
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
os.environ.setdefault("F5_TEST_RUN", "1")

import image_router as R
import ocr_reader

ok = []


def check(c, m):
    ok.append(bool(c)); print(("ok   " if c else "FAIL ") + m)


# Which kind of edit a phrase is: the model's read, bench/edit_intent_live.py.

calls, reads = [], iter([["УРОЖАЙ 2025"], ["УРОЖАЙ 2026", "облака"]])
R._image = types.SimpleNamespace(
    edit_image_with_firered=lambda ctx, p, instr, seed=None: calls.append((instr, seed)) or f"out{len(calls)}.png")
ocr_reader.read = lambda p: next(reads)
out, got, want = R.add_text_verified(None, "src.png", "добавь надпись «УРОЖАЙ 2026» в небо", seed=5)
check(out == "out2.png", f"the misspelt year is rejected, the second render delivered ({out})")
check(got == "УРОЖАЙ 2026" and want == "УРОЖАЙ 2026", f"reading reported ({got!r})")
check(len({s for _, s in calls}) == 2, "each try uses a different seed")
check("exactly" in calls[0][0] and "УРОЖАЙ 2026" in calls[0][0], "instruction restates the exact words")

calls.clear()
ocr_reader.read = lambda p: ["УРОЖАЙ 2025"]
out, got, want = R.add_text_verified(None, "src.png", "добавь надпись «УРОЖАЙ 2026»")
check(len(calls) == R.TEXT_ADD_TRIES, "never right -> every try was used")
check(out is None, "and with no source file to draw on, nothing is delivered")

# with a real source the exact words are drawn onto the ORIGINAL instead
import tempfile, text_overlay
from PIL import Image
src = os.path.join(tempfile.gettempdir(), "_text_add_src.png")
Image.new("RGB", (800, 500), (60, 120, 200)).save(src)
calls.clear()
R.OUTPUT_DIR = tempfile.gettempdir()
out2, got2, _ = R.add_text_verified(None, src, "добавь надпись «УРОЖАЙ 2026» в небо")
check(out2 and os.path.exists(out2) and "text-overlay" in out2, f"fallback draws the text ({out2})")
check(got2 == "УРОЖАЙ 2026", "and reports the exact words")
check(Image.open(out2).size == (800, 500), "same size as the source")
check(text_overlay.placement("подпись снизу") == "bottom" and text_overlay.placement("в небо") == "top",
      "placement from the wording")
note = R.TEXT_ADD_FAILED_NOTE.format(n=R.TEXT_ADD_TRIES, best=got, want=want)
check(note.startswith("[TOOL ERROR]") and "2025" in note, "the refusal says what was read")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
