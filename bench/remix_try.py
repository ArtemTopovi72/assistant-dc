"""    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/remix_try.py words song lyrics.txt | voice voice_song melody_song   (app stopped)"""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import live_tg_drive as D, remix
_b, ctx = D.build()
a = sys.argv[1:]
out = remix.remix_words(ctx, a[1], Path(a[2]).read_text(encoding="utf-8")) if a[0] == "words" else remix.remix_voice(ctx, a[1], a[2])
print("done", out)
