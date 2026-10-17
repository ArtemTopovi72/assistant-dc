"""«скажи голосом: …» with the chat's voice replies off came back as text only:
voice went out by the session setting alone, never by what the turn asked.

Run: venv/Scripts/python.exe tests/test_tg_voice_on_request.py
"""
import sys, os, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
from fakes import reads

T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_voicereq_"))

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

CID = 999961
SAY = "скажи голосом: привет, я кот-космонавт"
_READS = reads(read=lambda t: {"needs_tool": False, "speak_aloud": "голосом" in t})


class _Ctx:
    def __init__(self):
        import threading
        self.cancel_event = threading.Event()
        self.stage_callback = None
        self.session_memory = []
        self.pinned_facts = []
        self.memory_lock = threading.Lock()
        self.memory_text = lambda *a, **k: ""
        self.total_user_turns = 0
        self.last_image_path = ""
        self.last_image_prompt = ""
        self.tts_disabled = False
        for name, fn in _READS.items():
            setattr(self, name, fn)
    def set_stage(self, s):
        if self.stage_callback: self.stage_callback(s)


def run_turn(user_text, voice_on=False, tts_on=True):
    ctx = _Ctx()
    seen = {"voice": [], "text": [], "tts_disabled": None}

    class _Graph:
        def invoke(self, state, cfg=None):
            seen["tts_disabled"] = ctx.tts_disabled
            return {"final_answer": "Привет, я кот-космонавт!"}

    bot = T.TelegramBot("1:T", lambda: ctx, lambda: _Graph(),
                        lambda: {"messages": []}, silent_mode=True, tts_enabled_fn=lambda: tts_on)
    bot._send_text = lambda cid, text, **k: (seen["text"].append(text), 1)[1]
    bot._send_get_id = lambda cid, text, **k: 11
    bot._edit_text = lambda *a, **k: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._send_voice = lambda c, cid, text: (seen["voice"].append(text), True)[1]

    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    sess = bot._get_session(CID); sess.lang = "ru"; sess.voice_on = voice_on
    bot._store.put(sess)
    task = T._Task(task_id="t", chat_id=CID, user_text=user_text)
    bot._run_task_inner(task, bot._get_session(CID), bot._user_store.get(CID),
                        "U", "ru", ctx, _Graph())
    return seen


seen = run_turn(SAY)
check("asked to say it by voice: a voice note goes out with voice replies off", seen["voice"], str(seen))
check("and the graph is let to synthesize it", seen["tts_disabled"] is False, str(seen))
check("the text still goes out beside it", seen["text"], str(seen))

seen = run_turn("привет, как дела?")
check("an ordinary message with voice off gets no voice note", not seen["voice"], str(seen))
check("and synthesis stays off", seen["tts_disabled"] is True, str(seen))

seen = run_turn(SAY, tts_on=False)
check("with TTS off for the bot no voice is attempted", not seen["voice"] and seen["tts_disabled"] is True, str(seen))

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
