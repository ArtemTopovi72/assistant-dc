"""Regression: the vision proxy must always be a format the endpoint can decode.

BUG (the ".webp retry storm" in the logs): downscale_image_bytes short-circuited
on any image whose longest edge was already <= max_edge and returned the ORIGINAL
bytes. The analysis proxy is then wrapped by image_bytes_to_data_url, whose mime
defaults to "image/jpeg" — so a small .webp went out as webp bytes labelled JPEG,
the vision server could not decode it, and the retry ladder re-sent the identical
undecodable payload. Large webp hid the bug because the resize branch already
re-encoded to JPEG.

Run: venv/Scripts/python.exe tests/test_vision_proxy_format.py
"""
import io, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PIL import Image
from utils import downscale_image_bytes, image_bytes_to_data_url

ok = fail = 0

def check(label, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"PASS {label}")
    else:
        fail += 1
        print(f"FAIL {label} {extra}")


def encode(fmt, size, colour=(200, 90, 40)):
    im = Image.new("RGB", size, colour)
    buf = io.BytesIO()
    im.save(buf, fmt)
    return buf.getvalue()


def fmt_of(data):
    with Image.open(io.BytesIO(data)) as im:
        return (im.format or "").upper()


def dims_of(data):
    with Image.open(io.BytesIO(data)) as im:
        return im.size


# ── 1. the bug: a SMALL webp must be transcoded, not passed through ───────────
small_webp = encode("WEBP", (800, 600))
out = downscale_image_bytes(small_webp)
check("small webp is transcoded to JPEG", fmt_of(out) == "JPEG", fmt_of(out))
check("small webp keeps its pixel dimensions", dims_of(out) == (800, 600), dims_of(out))
check("the data URL mime now matches the bytes",
      image_bytes_to_data_url(out).startswith("data:image/jpeg;base64,"))

# ── 2. other unsafe formats are covered by the same rule ─────────────────────
for fmt in ("BMP", "TIFF", "GIF"):
    src = encode(fmt, (640, 480))
    got = downscale_image_bytes(src)
    check(f"small {fmt} is transcoded to JPEG", fmt_of(got) == "JPEG", fmt_of(got))

# ── 3. safe formats are still passed through byte-identically (no re-encode) ──
small_jpeg = encode("JPEG", (800, 600))
check("small JPEG is untouched", downscale_image_bytes(small_jpeg) == small_jpeg)
small_png = encode("PNG", (800, 600))
check("small PNG is untouched", downscale_image_bytes(small_png) == small_png)

# ── 4. the oversize path is unchanged: resize AND JPEG ───────────────────────
big_png = encode("PNG", (4000, 2000))
big_out = downscale_image_bytes(big_png)
check("oversize is still resized to the 1536 cap", max(dims_of(big_out)) == 1536,
      dims_of(big_out))
check("oversize is still JPEG", fmt_of(big_out) == "JPEG", fmt_of(big_out))
check("oversize keeps its aspect ratio", dims_of(big_out) == (1536, 768),
      dims_of(big_out))

big_webp = encode("WEBP", (3000, 3000))
bw = downscale_image_bytes(big_webp)
check("oversize webp: resized and JPEG",
      fmt_of(bw) == "JPEG" and max(dims_of(bw)) == 1536, f"{fmt_of(bw)} {dims_of(bw)}")

# ── 5. junk input still fails safe (returns the input, never raises) ─────────
junk = b"not an image at all"
check("junk bytes are returned unchanged, no exception",
      downscale_image_bytes(junk) == junk)
check("empty bytes are returned unchanged", downscale_image_bytes(b"") == b"")

# ── 6. every produced proxy is decodable — the property that actually failed ──
for fmt in ("WEBP", "BMP", "GIF", "TIFF", "PNG", "JPEG"):
    for size in ((300, 200), (2400, 1200)):
        proxy = downscale_image_bytes(encode(fmt, size))
        try:
            with Image.open(io.BytesIO(proxy)) as im:
                im.load()
                decodable = True
                as_jpeg_ok = (im.format or "").upper() in ("JPEG", "PNG")
        except Exception:
            decodable = as_jpeg_ok = False
        check(f"proxy from {fmt} {size[0]}x{size[1]} is decodable and safe",
              decodable and as_jpeg_ok)

print(f"\n{ok}/{ok + fail} checks passed")
sys.exit(1 if fail else 0)
