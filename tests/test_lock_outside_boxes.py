"""An edit is kept only inside its box (+pad); the rest of the frame is the original, byte for byte."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import compositing

d = tempfile.mkdtemp()
o, r = os.path.join(d, "o.png"), os.path.join(d, "r.png")
Image.new("RGB", (400, 400), (10, 20, 30)).save(o)
Image.new("RGB", (400, 400), (200, 100, 50)).save(r)
assert compositing.lock_outside_boxes(o, r, [[400, 400, 600, 600]])
im = Image.open(r)
assert im.getpixel((200, 200)) == (200, 100, 50), "edit lost inside the box"
assert im.getpixel((5, 5)) == (10, 20, 30), "far pixels must be the original"
assert not compositing.lock_outside_boxes(o, r, [])
print("ok")
