"""A picture's caption is part of what the picture says.

Live 16:57: a forwarded portrait with a quote over it («Никогда не спорьте …
— Марк Твен») was described as «рядом с ним виден текст на кириллице» -- the
user wanted to discuss the quote and the bot had not read it. The words are
taken in (what they say, whom they are attributed to), not recited in full.
Live LM Studio (vision); skipped without it, or without the picture, which is
kept out of the repo: runtime/live_media/twain_quote.jpg.
Run: venv/Scripts/python.exe tests/test_vision_reads_caption.py
"""
import os
import re
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT] + [os.path.join(ROOT, d) for d in ("core", "agent", "bot", "voice", "media", "imaging", "research", "knowledge", "services")]
import requests  # noqa: E402

PIC = os.path.join(ROOT, "runtime", "live_media", "twain_quote.jpg")
try:
    import config
    models = requests.get(config.LM_STUDIO_BASE + "/v1/models", timeout=5).json()["data"]
except Exception:
    models = []
if not models or not os.path.exists(PIC):
    print("SKIP  needs LM Studio and " + PIC)
    sys.exit(0)
import llm  # noqa: E402
from prompts import VISION_PROMPT  # noqa: E402
from utils import downscale_image_bytes  # noqa: E402


class Ctx:
    model_name = next((m["id"] for m in models if "gemma" in m["id"]), models[0]["id"])
    no_think = True
    reasoning_effort = None
    stop_event = None

    def is_cancelled(self):
        return False

    def set_stage(self, stage):
        pass


bad = 0
img = downscale_image_bytes(open(PIC, "rb").read())
for i in range(int(os.getenv("RUNS", "3"))):
    out = llm.analyze_image_with_llm(ctx=Ctx(), image_bytes=img, user_text="", system_prompt=VISION_PROMPT) or ""
    took_in = re.search(r"Твен|Twain", out) and re.search(r"спор|нож|argu|knife", out, re.I)
    vague = re.search(r"текст на кириллице|надпись на кириллице|Cyrillic text|some text", out, re.I)
    ok = bool(took_in) and not vague
    print(("PASS" if ok else "FAIL") + f"  run {i + 1}: the caption's meaning and author are in the description")
    if not ok:
        print("   " + out[:400].replace("\n", " | "))
    bad += not ok
sys.exit(1 if bad else 0)
