"""A forwarded video with no speech is retold from its storyboard.

Live 16:11: a 7-second кружок with no words, «📋 Краткое содержание» ->
«пришлите ТРАНСКРИПТ… у меня есть только раскадровка». The retelling prompt
promised a transcript; given a storyboard and no words the model asked for the
missing half instead of retelling what it saw. Silence is now decided where
speech is heard and reaches the retelling as «(без слов)». The offline part always runs; the live part needs LM Studio and is
skipped without it.
Run: venv/Scripts/python.exe tests/test_retell_silent_video.py
"""
import os
import re
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT] + [os.path.join(ROOT, d) for d in ("core", "agent", "bot", "voice", "media", "imaging", "research", "knowledge", "services")]
import tg_tasks  # noqa: E402

BOARD = ("👁 Раскадровка видео:\n"
         "0:00 — Крупный план лица молодого человека на фоне экрана.\n"
         "0:01 — Камера переводится на экран телевизора с ярким интерфейсом.\n"
         "0:02 — Экран телевизора, на котором видны красные и белые элементы меню.\n"
         "0:03 — Тот же кадр с экраном телевизора, изображение слегка смещается.\n"
         "0:04 — Крупный план экрана телевизора с различными иконками приложений.\n"
         "0:06 — Экран телевизора, на котором отображается заставка Disney.\n\n"
         "Итог: Видео представляет собой быструю съемку лица человека и переключение "
         "внимания на экран телевизора с меню приложений и логотипом Disney.")
bad = 0


def check(name, cond, extra=""):
    global bad
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   {extra}"))
    bad += not cond


# Silence is decided once, where speech is heard (_transcribe): a stray word
# ASR finds in a video is no speech, so every reader gets "" and the composer
# writes «(без слов)». Live 16:11: Whisper heard «Oh» in 7 silent seconds.
import audio  # noqa: E402
_heard = {"text": "Oh"}
audio.transcribe_audio_file = lambda *a, **k: _heard["text"]
audio.detect_media_language = lambda *a, **k: "en"
runner = tg_tasks.TaskRunnerMixin()
runner._dl_bytes = lambda fid: b"not really an mp4"
check("a stray «Oh» in a video is no speech", runner._transcribe(None, "f", "video") == "")
_heard["text"] = "смотри, я купил новый телевизор"
check("a video with real speech keeps it", runner._transcribe(None, "f", "video") == _heard["text"])
_heard["text"] = "Да"
check("a voice note's one word is real speech", runner._transcribe(None, "f", "voice") == "Да")
check("the retelling prompt reads the composer's no-speech marker",
      "(без слов)" in tg_tasks.retell_system("ru", True))
SAID = "(без слов)\n\n" + BOARD

import requests  # noqa: E402
try:
    import config      # the server llm talks to (run_all points it at a dead port)
    models = requests.get(config.LM_STUDIO_BASE + "/v1/models", timeout=5).json()["data"]
except Exception:
    models = []
if not models:
    print("SKIP  live retelling: LM Studio is not reachable")
    sys.exit(1 if bad else 0)
import llm  # noqa: E402


class Ctx:
    model_name = next((m["id"] for m in models if "gemma" in m["id"]), models[0]["id"])
    no_think = True
    reasoning_effort = None

    def is_cancelled(self):
        return False

    def set_stage(self, stage):
        pass


ASKS_FOR_TEXT = re.compile(r"транскрипт|пришлите|пришли(те)? текст|нет (слов|текста)|без (текста|транскрипта) (я )?не", re.I)
for i in range(int(os.getenv("RUNS", "3"))):
    out = llm.call_llm_simple(Ctx(), tg_tasks.retell_system("ru", True), SAID,
                              temperature=0.3, max_tokens=900, prefill="<think></think>") or ""
    check(f"live run {i + 1}: retold from the storyboard, no request for a transcript",
          bool(out.strip()) and not ASKS_FOR_TEXT.search(out) and re.search(r"телевиз|Disney|меню", out, re.I),
          out[:400].replace("\n", " | "))
sys.exit(1 if bad else 0)
