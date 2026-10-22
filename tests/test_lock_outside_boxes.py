"""An edit is kept inside its box (+pad) and wherever it drew past the box; the rest of the
frame is the original. The annotator's box held the lenses and the lock cut aviator glasses'
temple and turned sunglasses into a strip (2026-10-22); Klein's faint drift stays locked."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw
import compositing

d = tempfile.mkdtemp()
o, r = os.path.join(d, "o.png"), os.path.join(d, "r.png")
Image.new("RGB", (400, 400), (10, 20, 30)).save(o)
Image.new("RGB", (400, 400), (200, 100, 50)).save(r)
assert compositing.lock_outside_boxes(o, r, [[400, 400, 600, 600]])
im = Image.open(r)
assert im.getpixel((200, 200)) == (200, 100, 50), "edit lost inside the box"
assert im.getpixel((5, 5)) == (10, 20, 30), "a whole-frame re-render is not an edit: far pixels stay original"
assert not compositing.lock_outside_boxes(o, r, [])

# glasses: lenses in the box, a 3 px temple running out of it to the ear, faint drift elsewhere
skin = (200, 150, 130)
Image.new("RGB", (400, 400), skin).save(o)
res = Image.new("RGB", (400, 400), (212, 160, 138))          # drift: +12, under the threshold
dr = ImageDraw.Draw(res)
dr.rectangle([100, 180, 200, 230], fill=(20, 20, 25))          # lens
dr.line([(200, 200), (360, 215)], fill=(90, 90, 95), width=3)   # temple, past the box
res.save(r)
assert compositing.lock_outside_boxes(o, r, [[450, 250, 575, 500]], pad=0)
im = Image.open(r)
assert im.getpixel((330, 212))[0] < 140, "the temple drawn past the box was cut"
assert im.getpixel((150, 205)) == (20, 20, 25), "the lens was lost"
assert im.getpixel((50, 350)) == skin, "faint drift away from the edit must be the original"
print("ok")
