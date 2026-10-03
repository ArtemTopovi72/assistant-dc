"""Same relative crop from several images, upscaled and placed side by side.

    venv/Scripts/python bench/crops.py OUT.png x0 y0 x1 y1 img1 img2 ...
(x/y are fractions 0..1 of each image's own size, so different resolutions line up)
"""
import sys
from PIL import Image, ImageDraw

out, box, files = sys.argv[1], [float(v) for v in sys.argv[2:6]], sys.argv[6:]
tiles = []
for f in files:
    im = Image.open(f).convert("RGB")
    w, h = im.size
    c = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    s = 512 / max(c.size)
    c = c.resize((int(c.size[0] * s), int(c.size[1] * s)), Image.LANCZOS)
    ImageDraw.Draw(c).text((6, 6), f.replace("\\", "/").split("/")[-2] + "/" + f.split("\\")[-1].split("/")[-1],
                           fill=(255, 0, 255))
    tiles.append(c)
W = sum(t.size[0] for t in tiles) + 6 * (len(tiles) - 1)
H = max(t.size[1] for t in tiles)
sheet = Image.new("RGB", (W, H), "white")
x = 0
for t in tiles:
    sheet.paste(t, (x, 0))
    x += t.size[0] + 6
sheet.save(out)
print(out, sheet.size)
