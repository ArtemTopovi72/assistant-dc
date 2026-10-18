"""Contact sheet for bench/klein_vs_ideogram.py: rows = scenes, columns = variants, seconds in the label."""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("KOUT") or os.path.join(ROOT, "runtime", "klein_vs_ideogram")
T = int(os.environ.get("TILE", 384))
vs = [v for v in (sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(os.listdir(OUT))) if os.path.exists(os.path.join(OUT, v, "timings.json"))]
data = {v: json.load(open(os.path.join(OUT, v, "timings.json"), encoding="utf-8")) for v in vs}
names = [r["scene"] for r in next(iter(data.values()))]
font = ImageFont.truetype("arial.ttf", 18)
sheet = Image.new("RGB", (T * len(vs), (T + 26) * len(names)), "white")
d = ImageDraw.Draw(sheet)
for c, v in enumerate(vs):
    for r, n in enumerate(names):
        row = next((x for x in data[v] if x["scene"] == n), None)
        x, y = c * T, r * (T + 26)
        if row and "file" in row:
            sheet.paste(Image.open(row["file"]).convert("RGB").resize((T, T)), (x, y + 26))
            d.text((x + 4, y + 2), f"{v} | {n} | {row['seconds']}s", fill="black", font=font)
        else:
            d.text((x + 4, y + 2), f"{v} | {n} | ERR", fill="red", font=font)
p = os.path.join(OUT, "sheet.png")
sheet.save(p)
print(p, sheet.size)
