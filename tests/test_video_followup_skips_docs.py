"""A question about a just-forwarded VIDEO must not fall through to document
retrieval.

Live 2026-09-18 20:14 (chat 100000001): a video was forwarded and summarised,
then "Что ответить?" (no new photo, no image_id, no target_image -- a video
sheet isn't any of those) fell through the _about_a_picture gate in
tg_tasks._execute_task and got wrapped into the RAG prompt around 25 passages
of a completely UNRELATED indexed .md file ("...quant to solve agentic
tasks"). The resulting bloated, irrelevant context sent the model into a
non-terminating generation -- the task was still "running" 9+ minutes later.

Run: venv/Scripts/python.exe tests/test_video_followup_skips_docs.py
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_tasks as T, intent
# Whether it asks about the video is the model's read (agent/intent.py).
intent.STUB = lambda t: None if t == "привет" else {"is_question": True, "about_picture": True}

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

VIDEO_SHEET = os.path.join("runtime", "video_123", "sheet.jpg")
PHOTO_UPLOAD = os.path.join("runtime", "user_upload_1.jpg")

# A follow-up about the video on screen: must be treated as "about a
# picture" (bypasses the doc-RAG gate), same as the live incident.
FOLLOWUPS = [
    "Что ответить?",
    "что думаешь про это видео?",
    "заметил что-то странное на видео?",
    "what do you think?",
    "any flaws in the footage?",
]
for t in FOLLOWUPS:
    check("video follow-up bypasses docs: " + t,
          T._about_the_last_video(VIDEO_SHEET, t))

# Not a question / not video-shaped: no reason to force a bypass.
check("a bare greeting is not a video follow-up",
      not T._about_the_last_video(VIDEO_SHEET, "привет"))

# No video on screen at all: never bypasses on this signal alone.
check("empty last_image_path never bypasses",
      not T._about_the_last_video("", "Что ответить?"))

# The signal is video-SHEET specific: an ordinary photo upload must not
# trigger it (that path is already covered by target_image/image_id).
check("a plain photo upload path does not count as a video",
      not T._about_the_last_video(PHOTO_UPLOAD, "что думаешь?"))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
