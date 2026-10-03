"""Generate the Assistant app icon: a microphone on a blue->teal gradient.

Writes assets/assistant.ico (9 sizes, 16..256 for a crisp taskbar) and a PNG
preview. Re-run after editing to regenerate.
"""
import os
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

S = 256
TOP = (0x2b, 0x6c, 0xf0)   # accent blue
BOT = (0x00, 0xd3, 0xa7)   # accent teal

# diagonal-ish vertical gradient
grad = Image.new("RGB", (S, S))
gd = ImageDraw.Draw(grad)
for y in range(S):
    t = y / (S - 1)
    c = tuple(int(TOP[i] + (BOT[i] - TOP[i]) * t) for i in range(3))
    gd.line([(0, y), (S, y)], fill=c)

# rounded-square mask
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=54, fill=255)

icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
icon.paste(grad, (0, 0), mask)

d = ImageDraw.Draw(icon)
cx = S // 2
W = (255, 255, 255, 255)
# mic capsule (head)
d.rounded_rectangle([cx - 36, 56, cx + 36, 150], radius=36, fill=W)
# holder (U arc under the head)
d.arc([cx - 56, 72, cx + 56, 188], start=18, end=162, fill=W, width=14)
# stand
d.rounded_rectangle([cx - 7, 182, cx + 7, 212], radius=4, fill=W)
# base
d.rounded_rectangle([cx - 46, 210, cx + 46, 224], radius=7, fill=W)

SIZES = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40),
         (48, 48), (64, 64), (128, 128), (256, 256)]
ico_path = os.path.join(ASSETS, "assistant.ico")
icon.save(ico_path, format="ICO", sizes=SIZES)
icon.save(os.path.join(ASSETS, "assistant_preview.png"), format="PNG")
print("wrote", ico_path, "with sizes:", [s[0] for s in SIZES])
