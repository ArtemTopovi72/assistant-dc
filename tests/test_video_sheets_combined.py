"""A forwarded batch of video notes: the chat's picture holds EVERY clip.
Only the last clip's sheet used to stay, so «look again» saw one of nine.
Run: venv/Scripts/python.exe tests/test_video_sheets_combined.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for d in ("media", "core"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d))
from pathlib import Path
from PIL import Image
import video_look

tmp = Path(tempfile.mkdtemp())
parts = []
for i, c in enumerate([(200, 0, 0), (0, 200, 0), (0, 0, 200)]):
    p = tmp / f"s{i}.jpg"
    Image.new("RGB", (1280, 960), c).save(p)
    parts.append((f"видео {i + 1} · Борис", str(p)))
out = video_look.combine_sheets(parts, tmp / "video_x" / "sheet.jpg")
im = Image.open(out).convert("RGB")
seen = [im.getpixel((480, y)) for y in (52 + 360, 2 * 52 + 720 + 360, 3 * 52 + 1440 + 360)]
ok = im.size == (960, 3 * (52 + 720)) and seen[0][0] > 150 and seen[1][1] > 150 and seen[2][2] > 150
print(("PASS" if ok else "FAIL") + "  all three clips stacked in order under their bars", im.size, seen)
sys.exit(0 if ok else 1)
