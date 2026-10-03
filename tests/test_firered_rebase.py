"""The 4th FireRed edit in a row renders from the ORIGINAL with every
instruction so far (edit-on-edit drift); tiles and reference edits never do."""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "imaging", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ["F5_TEST_RUN"] = "1"
import image as I

tmp = tempfile.mkdtemp()
calls = []
n = [0]
def once(ctx, src, instr, **kw):
    calls.append((os.path.basename(src), instr))
    n[0] += 1
    out = os.path.join(tmp, f"r{n[0]}.png")
    open(out, "wb").write(b"x")
    return out
I._edit_image_with_firered_once = once

orig = os.path.join(tmp, "orig.png"); open(orig, "wb").write(b"x")
cur = orig
for i, ins in enumerate(["blue shirt", "winter", "no hat", "add a dog"]):
    cur = I.edit_image_with_firered(None, cur, ins)
assert calls[2] == ("r2.png", "no hat"), calls
assert calls[3] == ("orig.png", "blue shirt; then winter; then no hat; then add a dog"), calls[3]
print("ok the 4th edit in a row starts from the original with all instructions")

calls.clear()
I.edit_image_with_firered(None, cur, "red sky", save_prefix="tile")
assert calls[0][0] == os.path.basename(cur) and calls[0][1] == "red sky", calls
print("ok a contained tile edit is never rebased")
