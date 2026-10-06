"""▶️ Continue video: the start frame and the clip's tail are taken, the tool gets the tail as <Video 1>,
and the result is the original joined with its continuation.

Real ffmpeg on a synthetic 4 s clip (test pattern, a red block appears at 3.3 s -- the coat taken off --
silence for 2 s then a tone). The flow: button -> video -> "what happens next" -> the animate request -> generate_video with the
tail as its reference video -> the joined clip.
"""
import os, sys, time, threading, tempfile, subprocess
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = tempfile.mkdtemp(prefix="contvid_")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
T.redirect_data_dir(_DATA)
import graph as graph_mod

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

CLIP = os.path.join(_DATA, "clip.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=300:duration=4",
                "-vf", "drawbox=x=100:y=80:w=120:h=80:color=red:t=fill:enable='gte(t,3.3)'",
                "-af", "volume=0:enable='lt(t,2)'", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", CLIP], check=True)

import tg_continue as C
import video_look
from PIL import Image
import numpy as np

frame = os.path.join(_DATA, "last.jpg")
check("a start frame is taken", C.seed_frame(CLIP, frame) and os.path.exists(frame))
# Live 10-06: the frame came from 2 s before the end; the coat taken off in those 2 s was back on.
px = np.asarray(Image.open(frame).convert("RGB"), dtype=np.int16)[100:140, 130:190].mean(axis=(0, 1))
check("the start frame shows the clip's FINAL state (what changed near the end stays changed)",
      px[0] > 180 and px[1] < 80 and px[2] < 80, px.tolist())

tail = os.path.join(_DATA, "tail.mp4")
import video as V
check("the tail is cut with its sound", C.cut_tail(CLIP, tail)
      and 2.0 < V.probe(tail)["seconds"] < 3.0 and V.probe(tail)["has_audio"], V.probe(tail))

# the join: original + continuation, crossfaded, first frames of the new part dropped
NEW = os.path.join(_DATA, "new.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=3",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", NEW], check=True)      # a continuation WITHOUT sound
joined = V.join_continuation(CLIP, NEW)
pj = V.probe(joined) if joined else {}
check("original + silent continuation are joined, with sound", bool(joined) and pj.get("has_audio"), pj)
check("...the length is both parts minus the crossfade and the dropped frames",
      abs(pj.get("seconds", 0) - (4.0 + 3.0 - 0.25 - 2 / 24)) < 0.35, pj)

# ---- the flow ----
INVOKES = []
class FakeCtx:
    def __init__(self):
        self.session_memory = []; self.pinned_facts = []
        self.cancel_event = threading.Event(); self.last_image_path = ""
        self.last_image_prompt = ""; self.stage_callback = None
        self.total_user_turns = 0; self.tts_disabled = True
        self.memory_lock = threading.Lock()
    def memory_text(self): return ""
    def set_stage(self, s): pass
class StubGraph:
    def __init__(self, ctx): self.ctx = ctx
    def invoke(self, base):
        INVOKES.append(base.get("user_input", ""))
        return {"final_answer": "echo", "messages": base.get("messages", [])}
graph_mod.build_graph = lambda ctx: StubGraph(ctx)
SHARED = FakeCtx()
bot = T.TelegramBot("123:TEST", lambda: SHARED, lambda: object(), lambda: {"messages": []}, silent_mode=True)
bot._backend = T.InMemoryBackend(); bot.sent = []
bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t)), 1)[1]
bot._send_get_id = lambda cid, t, **kw: (bot.sent.append((cid, t)), len(bot.sent))[1]
bot._edit_text = lambda *a, **kw: None; bot._delete = lambda *a, **kw: None
bot._api_post = lambda *a, **k: {}; bot._api_get = lambda *a, **k: {}
bot._activity.log = lambda *a, **k: None
bot._running = True
threading.Thread(target=bot._consumer_loop, daemon=True).start()

CID = 9_300_088
bot._user_store.put(T._User(chat_id=CID, name="Cv", status="approved", is_admin=False))
sess = bot._get_session(CID); sess.clear_context(); sess.lang = "ru"; sess.lang_chosen = True; sess.reg_state = ""; bot._store.put(sess)
bot._dl_bytes = lambda fid, *a, **k: open(CLIP, "rb").read()
_seq = [43_000_000]
def _nid(): _seq[0] += 1; return _seq[0]
def m(**kw):
    d = {"message_id": _nid(), "chat": {"id": CID, "type": "private"}, "from": {"id": CID}, "date": int(time.time())}
    d.update(kw); return {"update_id": _nid(), "message": d}
def wait(pred, t=10):
    e = time.monotonic() + t
    while time.monotonic() < e:
        if pred(): return True
        time.sleep(0.1)
    return pred()

bot._dispatch(m(text="▶️ Продолжить видео"))
check("the button asks for a video",
      wait(lambda: bot._get_session(CID).continue_state == "want_video"), bot.sent[-2:])
bot._dispatch(m(video={"file_id": "VD", "duration": 4, "mime_type": "video/mp4"}))
check("the video is taken: now asks what happens next",
      wait(lambda: bot._get_session(CID).continue_state == "want_text"), bot.sent[-2:])
s_ = bot._get_session(CID)
check("its tail waits as the reference video", os.path.exists(s_.continue_tail) and os.path.exists(s_.continue_src))
check("its start frame is the current image", bool(s_.target_image))
bot._dispatch(m(text="они выходят из комнаты и смеются"))
check("the text becomes the animate request",
      wait(lambda: any(t.startswith("animate this photo: они выходят") for t in INVOKES), 15), INVOKES[-2:])
check("the state is cleared", bot._get_session(CID).continue_state == "")

# ---- the tool: the tail becomes <Video 1>, the result is the joined clip ----
import tool_image_handlers as H
import tools as _T
SEEN = {}
def fake_gen(ctx, description, **kw):
    SEEN.update(description=description, **kw)
    return {"path": NEW, "status": "success", "seconds": 3.0, "width": 320, "height": 240}
V.generate_video = fake_gen
V.engine_available = lambda ctx=None: (True, "")
H.asks_for_video = lambda ctx, state: True
_T._render_budget_exhausted = lambda *a, **k: None
H._current_image_paths = lambda ctx, state: [frame]
H._remember = lambda *a, **k: None
class TCtx:
    anim_voices = []; voice_choice = "default"; voice_ref = ""
    def __init__(self): self.continue_tail = tail; self.continue_src = CLIP
    def set_stage(self, *a): pass
out = H._handle_generate_video(TCtx(), {}, {"description": "they leave the room laughing"})
check("generate_video gets the tail as its reference video", SEEN.get("videos") == [tail], SEEN.get("videos"))
check("...and the picture, no separate voices", SEEN.get("images") == [frame] and not SEEN.get("audios"), SEEN)
check("...with a continue-without-a-cut prompt carrying the user's words",
      SEEN["description"].startswith("Continue the shot of <Video 1>") and "laughing" in SEEN["description"])
st = {}
H._handle_generate_video(TCtx(), st, {"description": "they leave"})
check("the delivered clip is the joined one (longer than the new part)",
      st.get("video_path") and V.probe(st["video_path"])["seconds"] > 5.5, st)

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
