"""The delivery helpers stranded scratch files on every failure path.

Both build their output with NamedTemporaryFile(delete=False), and both had
return paths that walked away from the file they had just created:

  * make_thumbnail: ffmpeg writes nothing (a codec it cannot decode, a timeout),
    the size check fails, and the function returns None — leaving a 0-byte
    temp file behind. The caller only deletes a thumbnail it was GIVEN.
  * fit_for_telegram: the re-encode finishes but is STILL over the cap, so the
    function returns the original path. tg_bot deletes send_path only when it
    differs from path, so the near-cap re-encode (up to 50MB) is stranded — once
    per oversize clip, forever.

Neither is visible in a chat: the video still goes out. It just quietly fills
the disk. ffmpeg is stubbed here, so this runs anywhere and never renders.

Run: venv/Scripts/python.exe tests/test_video_delivery_tempfiles.py
"""
import os
import sys
import glob
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import video as V

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


TMPDIR = tempfile.gettempdir()

def _scratch(prefix):
    return set(glob.glob(os.path.join(TMPDIR, prefix + "*")))


def _make_file(path, size):
    with open(path, "wb") as fh:
        fh.write(b"\x00" * size)
    return path


WORK = tempfile.mkdtemp(prefix="viddeliv_")
SRC = _make_file(os.path.join(WORK, "render.mp4"), 4096)

_real_run, _real_which = V.subprocess.run, V.shutil.which
V.shutil.which = lambda name: "ffmpeg-stub"          # never touches real ffmpeg


class _Run:
    """Stand-in for subprocess.run that writes whatever the scenario needs."""
    def __init__(self, write_bytes=None, raise_exc=None):
        self.write_bytes, self.raise_exc = write_bytes, raise_exc
    def __call__(self, argv, **kw):
        if self.raise_exc:
            raise self.raise_exc
        out = argv[-1]                                # both helpers put it last
        if self.write_bytes is not None:
            _make_file(out, self.write_bytes)
        class _P: returncode = 0; stdout = ""; stderr = ""
        return _P()


try:
    print("=" * 70)
    print("1. make_thumbnail cleans up when ffmpeg produces nothing")
    print("=" * 70)

    before = _scratch("vidthumb_")
    V.subprocess.run = _Run(write_bytes=0)            # ffmpeg wrote an empty file
    got = V.make_thumbnail(SRC)
    leaked = _scratch("vidthumb_") - before
    check("returns None when the frame is empty", got is None, got)
    check("no scratch thumbnail is left behind", not leaked, sorted(leaked))

    before = _scratch("vidthumb_")
    V.subprocess.run = _Run(raise_exc=RuntimeError("ffmpeg exploded"))
    got = V.make_thumbnail(SRC)
    leaked = _scratch("vidthumb_") - before
    check("returns None when ffmpeg raises", got is None, got)
    check("no scratch thumbnail after an exception", not leaked, sorted(leaked))

    before = _scratch("vidthumb_")
    V.subprocess.run = _Run(write_bytes=256)          # the good path
    got = V.make_thumbnail(SRC)
    check("a real thumbnail IS returned", bool(got) and os.path.exists(got), got)
    check("...and it is the file the caller must clean up",
          got in (_scratch("vidthumb_") - before), got)
    if got:
        os.remove(got)

    print()
    print("=" * 70)
    print("2. fit_for_telegram cleans up a re-encode that still does not fit")
    print("=" * 70)

    CAP = 1024
    BIG = _make_file(os.path.join(WORK, "big.mp4"), CAP * 4)

    before = _scratch("vidtg_")
    V.subprocess.run = _Run(write_bytes=CAP * 3)      # smaller, but still over
    out = V.fit_for_telegram(BIG, max_bytes=CAP)
    leaked = _scratch("vidtg_") - before
    check("falls back to the ORIGINAL when the re-encode does not fit",
          out == BIG, out)
    check("the over-cap re-encode is not stranded", not leaked, sorted(leaked))

    before = _scratch("vidtg_")
    V.subprocess.run = _Run(raise_exc=RuntimeError("encoder died"))
    out = V.fit_for_telegram(BIG, max_bytes=CAP)
    leaked = _scratch("vidtg_") - before
    check("falls back to the ORIGINAL when ffmpeg raises", out == BIG, out)
    check("nothing stranded after an exception", not leaked, sorted(leaked))

    before = _scratch("vidtg_")
    V.subprocess.run = _Run(write_bytes=CAP // 2)     # the good path
    out = V.fit_for_telegram(BIG, max_bytes=CAP)
    check("a fitting re-encode IS returned", out != BIG and os.path.exists(out), out)
    check("...and it is the temp file the caller deletes",
          out in (_scratch("vidtg_") - before), out)
    # The source render is the USER'S file and must survive being re-encoded.
    check("the original render was not touched", os.path.exists(BIG))
    if out != BIG:
        os.remove(out)

    print()
    print("=" * 70)
    print("3. An under-cap clip is passed straight through, untouched")
    print("=" * 70)
    before = _scratch("vidtg_")
    V.subprocess.run = _Run(write_bytes=10)
    out = V.fit_for_telegram(SRC, max_bytes=CAP * 100)
    check("returns the same path", out == SRC, out)
    check("no re-encode was created", not (_scratch("vidtg_") - before))

    print()
    print("=" * 70)
    print("4. Adopting a clip never overwrites one we already kept")
    print("=" * 70)
    # ComfyUI's 00001_ counter comes from ITS output tree; clean that tree and the
    # numbering restarts, so a second render can arrive with a basename we have
    # already stored. A render is minutes of GPU — it must not be destroyed.
    COMFY_OUT = tempfile.mkdtemp(prefix="comfyout_")
    KEEP = tempfile.mkdtemp(prefix="ourout_")
    _real_outdir = V.OUTPUT_DIR
    V.OUTPUT_DIR = KEEP
    try:
        first = _make_file(os.path.join(COMFY_OUT, "h3_00001_.mp4"), 111)
        a = V._adopt_output(first)
        check("the first clip is adopted", os.path.exists(a) and a.startswith(KEEP), a)

        second = _make_file(os.path.join(COMFY_OUT, "h3_00001_.mp4"), 222)
        b = V._adopt_output(second)
        check("a same-named second clip gets its own path", b != a, (a, b))
        check("...and both survive", os.path.exists(a) and os.path.exists(b), (a, b))
        check("the FIRST clip's bytes were not overwritten",
              os.path.getsize(a) == 111, os.path.getsize(a))
        check("the second clip has the new bytes", os.path.getsize(b) == 222,
              os.path.getsize(b))

        third = _make_file(os.path.join(COMFY_OUT, "h3_00001_.mp4"), 333)
        c = V._adopt_output(third)
        check("a third collision keeps counting", c not in (a, b) and os.path.exists(c), c)
        check("...with all three intact",
              [os.path.getsize(p) for p in (a, b, c)] == [111, 222, 333],
              [os.path.getsize(p) for p in (a, b, c)])

        # A clip already inside our own directory must be used in place, not
        # duplicated on every call.
        inplace = _make_file(os.path.join(KEEP, "already_here.mp4"), 44)
        check("a clip already in OUTPUT_DIR is returned as-is",
              V._adopt_output(inplace) == inplace)
    finally:
        V.OUTPUT_DIR = _real_outdir

finally:
    V.subprocess.run, V.shutil.which = _real_run, _real_which

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
