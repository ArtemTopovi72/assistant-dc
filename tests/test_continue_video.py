"""▶️ Continue video: the start frame and the clip's tail are taken, the tool gets the tail as <Video 1>,
and the result is the original joined with its continuation.

Real ffmpeg on a synthetic 4 s clip (test pattern, a red block appears at 3.3 s -- the coat taken off --
silence for 2 s then a tone). The flow: button -> video -> "what happens next" -> the animate request -> generate_video with the
tail as its reference video -> the joined clip.
"""
import os, sys, time, threading, tempfile, subprocess
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ.setdefault("VIDEO_AUTO_MAX_PARTS", "99")   # long chains are tested here; the 2-part cap has its own test
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = tempfile.mkdtemp(prefix="contvid_")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
T.redirect_data_dir(_DATA)

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
# The start frame comes from the very end of the clip, not from ~2 s earlier.
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
PUSHED = []
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
SHARED = FakeCtx()
bot = T.TelegramBot("123:TEST", lambda: SHARED, lambda: object(), lambda: {"messages": []}, silent_mode=True)
bot.graph_builder = lambda ctx: StubGraph(ctx)
bot._backend = T.InMemoryBackend(); bot.sent = []
_orig_push = bot._backend.push
def _rec(t):
    PUSHED.append(t)
    return _orig_push(t)
bot._backend.push = _rec
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
# A picture the chat delivered EARLIER must not become the continuation's
# target: the raw text consumes sess.target_image at its enqueue and the task
# re-resolved to sess.turn_image -- a stale render in the debug chat, a frame
# sheet in the owner's, and the agent wrote the continuation about the wrong
# scene (live 2026-11-02).
_stale = os.path.join(_DATA, "stale.png")
Image.new("RGB", (64, 64), (200, 30, 30)).save(_stale)
_stale_id = T._log_image(bot._get_session(CID), _stale, msg_id=7, label="старая картинка", src="bot")
sess = bot._get_session(CID); sess.turn_image = _stale_id; bot._store.put(sess)
bot._dispatch(m(video={"file_id": "VD", "duration": 4, "mime_type": "video/mp4"}))
check("the video is taken: now asks what happens next",
      wait(lambda: bot._get_session(CID).continue_state == "want_text"), bot.sent[-2:])
s_ = bot._get_session(CID)
check("its tail waits as the reference video", os.path.exists(s_.continue_tail) and os.path.exists(s_.continue_src))
check("its start frame is the current image", bool(s_.target_image))
_seed_id = s_.target_image
# A second clip (a кружок of «the guy with the glove») is a new person, never a new
# start: it replaced the grandma's GIF and the continuation lost her (live 2026-10-12).
_src0 = s_.continue_src
bot._dispatch(m(video_note={"file_id": "GUY", "duration": 4}))
check("a second video while waiting for words joins as a new person",
      wait(lambda: len(bot._get_session(CID).continue_people or []) == 1), bot.sent[-2:])
s_ = bot._get_session(CID)
check("...the first clip stays the one continued",
      s_.continue_src == _src0 and s_.continue_state == "want_text"
      and os.path.exists(s_.continue_people[0]), (s_.continue_src, s_.continue_people))
bot._dispatch(m(text="они выходят из комнаты и смеются"))
check("the text becomes the animate request",
      wait(lambda: any(t.startswith("continue this video: они выходят") for t in INVOKES), 15), INVOKES[-2:])
check("the state is cleared", bot._get_session(CID).continue_state == "")
check("the task carries the clip's start frame, not the chat's stale picture",
      bool(PUSHED) and PUSHED[-1].image_id == _seed_id and PUSHED[-1].image_id != _stale_id,
      [(t.image_id, _seed_id, _stale_id) for t in PUSHED])

# ---- the tool: the tail becomes <Video 1>, the result is the joined clip ----
import tool_image_handlers as H
import types
from fakes import Over
# What the handler is handed on each TCtx: the video engine's fakes (_E), the handlers'
# own helpers (_DEPS) and the text model (_TM[0], None = the real one).
_E, _DEPS, _TM = {}, {}, [None]
SEEN = {}
def fake_gen(ctx, description, **kw):
    SEEN.update(description=description, **kw)
    return {"path": NEW, "status": "success", "seconds": 3.0, "width": 320, "height": 240}
_E["generate_video"] = fake_gen
_E["engine_available"] = lambda ctx=None: (True, "")
_E["motion_context_on"] = lambda: False             # the reference-tail path first
_DEPS["asks_for_video"] = lambda ctx, state: True
_DEPS["_render_budget_exhausted"] = lambda *a, **k: None
_DEPS["_current_image_paths"] = lambda ctx, state: [frame]
_DEPS["_remember"] = lambda *a, **k: None
class TCtx:
    anim_voices = []; voice_choice = "default"; voice_ref = ""
    def __init__(self):
        self.continue_tail = tail; self.continue_src = CLIP
        # the suite's stand-ins, as they are when the context is made
        self.video_engine = Over(V, **_E)
        self.deps = types.SimpleNamespace(**_DEPS)
        self.text_model = _TM[0]
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
check("the delivered duration describes the joined clip",
      abs(st.get("video_seconds", 0) - V.probe(st["video_path"])["seconds"]) < 0.1, st)

# «…орёт э, это уже есть на видео, а потом…»: a plan part the user says is already in
# the clip is not rendered again (owner, 2026-10-12: 6 minutes for a repeat).
CALLS = []
_E["generate_video"] = lambda ctx, description, **kw: (CALLS.append(description), fake_gen(ctx, description, **kw))[1]
# what is already there, then how many leading sentences repeat it (one: the scream)
_TM[0] = lambda ctx, system, user, **k: (
    "она орёт" if "ALREADY" in system else "1" if "trim a video" in system else "")
_script = ("The old woman touches the things in the cabinet, jerks her hand back and screams eh at "
           "the top of her voice, her mouth wide open in shock for a long moment. Then the cabinet "
           "door slowly swings open with a creak while the young man inside waits in the dark. "
           "Sound: her scream screams on. The young man with a blue glove on his fist steps out of "
           "the dark, walks up to her and stops right in front of her, staring at her face. The "
           "camera glides slowly across the wardrobe and the clothes hanging in it, the light soft.")
H._handle_generate_video(TCtx(), {"user_input_original": "она орёт, это уже есть на видео, а потом выходит парень"},
                         {"description": _script})
_all = " ".join(CALLS)
check("the part already in the clip is not rendered; the door (its sound line echoing the scream) and the guy are",
      CALLS and "jerks her hand" not in _all and "swings open" in _all and "glove" in _all, CALLS)

# A new person rides on EVERY part: given to part 1 only, the guy from the кружок walked
# out in part 2 as a stranger (live 2026-10-12).
GUY = os.path.join(_DATA, "guy.jpg")
Image.new("RGB", (64, 64), "blue").save(GUY)
IMGS = []
_E["generate_video"] = lambda ctx, description, **kw: (IMGS.append(list(kw.get("images") or [])),
                                                   fake_gen(ctx, description, **kw))[1]
_TM[0] = lambda *a, **k: "NONE"
class PCtx(TCtx):
    def __init__(self):
        super().__init__(); self.continue_people = [GUY]
H._handle_generate_video(PCtx(), {}, {"description": _script})
check("every part of a continuation gets the new person's picture",
      len(IMGS) > 1 and all(GUY in imgs for imgs in IMGS), IMGS)
_E["generate_video"] = fake_gen
_TM[0] = None
_E["generate_video"] = fake_gen

# Pinned frames on: the tail is pinned into the new clip's head, not a reference
_E["motion_context_on"] = lambda: True
SEEN.clear()
H._handle_generate_video(TCtx(), {}, {"description": "they leave the room laughing"})
check("pinned: the tail goes in as context_video", SEEN.get("context_video") == tail, SEEN)
check("pinned: no reference video or picture", not SEEN.get("videos") and not SEEN.get("images"), SEEN)
check("pinned: the no-cut prompt without <Video 1>",
      SEEN["description"].startswith(V.CONTINUE_CTX_PREFIX) and "<Video 1>" not in SEEN["description"])

# Static-screen continuation (owner, 2026-11-02): «он говорит» stays ON the clip's
# final frame — H3's first+last-frame mode with first = last = that frame, no
# reference tail, anchored through the talking-head machinery.
_e_marker = None
_E["motion_context_on"] = lambda: False
_E["static_continuation"] = lambda ctx, d: True
REVOICED = []
_E["revoice"] = lambda ctx, clip, voice: (REVOICED.append((clip, voice)), clip)[1]
_E["voice_sample"] = lambda tail: tail
SEEN.clear()
TC = TCtx()
H._handle_generate_video(TC, {}, {"description": 'he looks into the camera and says "the line"'})
check("static: first = last = the clip's final frame (FL2VA)",
      SEEN.get("images") == [frame, frame], SEEN.get("images"))
check("static: no reference tail or audio rides",
      not SEEN.get("videos") and not SEEN.get("audios"), (SEEN.get("videos"), SEEN.get("audios")))
check("static: the talking-head machinery anchors it (keyframe)",
      SEEN.get("keyframe") is True, SEEN.get("keyframe"))
check("static: duration comes from the original script before camera instructions",
      SEEN.get('seconds', 0) >= 5.0, SEEN.get('seconds'))
check("static: the camera is told to hold the framing",
      "locked camera" in SEEN.get("description", "")
      and "no zoom" in SEEN.get("description", "")
      and "can move naturally" in SEEN.get("description", ""),
      SEEN.get("description", "")[:220])
check("static: the continuation prompt keeps the user's words",
      "says" in SEEN.get("description", ""), SEEN.get("description", "")[:200])
check("static: does not override the requested speech language", "said once, clearly, in Russian" not in SEEN.get("description", ""), SEEN.get("description", ""))
check("static: the new part uses the full clip for its speaker reference",
      bool(REVOICED) and REVOICED[0][1] == CLIP, REVOICED)
_E["overlay_drift"] = lambda path: V.OVERLAY_DRIFT_MAX + 1
_E["eye_drift"] = lambda *a, **k: 0.0
_scene_reply = H._handle_generate_video(TCtx(), {}, {"description": 'he says "the line"'})
check("static: changed background is disclosed even when the face check passes",
      "framing or appearance may have changed" in _scene_reply, _scene_reply)
_E.pop("overlay_drift"); _E.pop("eye_drift")
_E['estimate_seconds'] = lambda text, **kw: 10.0 if text.startswith(V.STATIC_CONTINUE_PREFIX) else 6.0
SEEN.clear()
H._handle_generate_video(TCtx(), {}, {"description": 'he says "the line"'})
check("static: render receives the unadorned script duration", SEEN.get('seconds') == 6.0, SEEN)
_E.pop('estimate_seconds')
_flash_calls = []
_E['generate_video'] = lambda ctx, description, **kw: (_flash_calls.append(1), fake_gen(ctx, description, **kw))[1]
_E['new_bright_region'] = lambda *a: len(_flash_calls) == 1
_E['eye_drift'] = lambda *a, **k: 0.0
class FlashCheckCtx(TCtx):
    speech_ok = staticmethod(lambda ctx, clip, lines: True)
    yes_no = staticmethod(lambda *a, **k: False)
_flash_reply = H._handle_generate_video(FlashCheckCtx(), {}, {"description": 'he says "the line"'})
check("a possible flash is retried and a clean take is selected", len(_flash_calls) == 2, _flash_calls)
_E['new_bright_region'] = lambda *a: True
_flash_calls.clear()
_flash_reply = H._handle_generate_video(FlashCheckCtx(), {}, {"description": 'he says "the line"'})
check("persistent possible flashes are disclosed", len(_flash_calls) == 3 and 'possible visual artifact' in _flash_reply, _flash_reply)
class RequestedFlashCtx(FlashCheckCtx):
    yes_no = staticmethod(lambda *a, **k: True)
_flash_calls.clear()
_flash_reply = H._handle_generate_video(RequestedFlashCtx(), {}, {"description": 'he says "the line" under flashing lights'})
check("requested bright effects are not treated as artifacts", len(_flash_calls) == 1 and 'possible visual artifact' not in _flash_reply, _flash_reply)
_E.pop('new_bright_region'); _E.pop('eye_drift'); _E['generate_video'] = fake_gen
SEEN.clear()
H._handle_generate_video(TCtx(), {"user_input_original": 'Мужчина говорит «то вам очко порвут»'},
                         {"description": 'The man says "то вам очко порют"'})
check("continuation restores the user's exact dictated line",
      "то вам очко порвут" in SEEN.get("description", "") and "порют" not in SEEN.get("description", ""), SEEN)
class UnclearSpeechCtx(TCtx):
    speech_ok = staticmethod(lambda ctx, clip, lines: False)

_unclear_reply = H._handle_generate_video(UnclearSpeechCtx(), {}, {"description": 'he says "the line"'})
check("unverified speech is disclosed", "could not confirm the requested line" in _unclear_reply, _unclear_reply)
class UnavailableSpeechCheckCtx(TCtx):
    speech_ok = staticmethod(lambda ctx, clip, lines: None)

_unknown_reply = H._handle_generate_video(UnavailableSpeechCheckCtx(), {}, {"description": 'he says "the line"'})
check("failed speech check is disclosed instead of claimed successful",
      "could not confirm the requested line" in _unknown_reply, _unknown_reply)
class RevoiceLostSpeechCtx(TCtx):
    def __init__(self):
        super().__init__(); self.checks = 0
    def speech_ok(self, ctx, clip, lines):
        self.checks += 1
        return self.checks == 1
_revoice_check = RevoiceLostSpeechCtx()
_revoice_reply = H._handle_generate_video(_revoice_check, {}, {"description": 'he says "the line"'})
check("speech is checked again after voice conversion",
      _revoice_check.checks >= 2 and "could not confirm the requested line" in _revoice_reply,
      _revoice_reply)
_E["eye_drift"] = lambda *a, **k: 5.0
class VisualDriftCtx(TCtx):
    speech_ok = staticmethod(lambda ctx, clip, lines: True)
_drift_reply = H._handle_generate_video(VisualDriftCtx(), {}, {"description": 'he says "the line"'})
check("visual drift is disclosed", "visual drift in the selected take" in _drift_reply, _drift_reply)
_E.pop("eye_drift")
_E["revoice"] = lambda ctx, clip, voice: None
_failed_state = {}
_failed_reply = H._handle_generate_video(TCtx(), _failed_state, {"description": 'he says "the line"'})
check("static: failed voice conversion is not reported as a finished matched-voice clip",
      "[TOOL ERROR]" in _failed_reply and not _failed_state.get("video_path"), _failed_reply)
_E["revoice"] = lambda ctx, clip, voice: clip
_attempts = []
def _cancel_second(ctx, description, **kwargs):
    _attempts.append(1)
    return fake_gen(ctx, description, **kwargs) if len(_attempts) == 1 else {"status": "cancelled"}
_E["generate_video"] = _cancel_second
_cancel_state = {}
_cancelled_reply = H._handle_generate_video(UnclearSpeechCtx(), _cancel_state, {"description": 'he says "the line"'})
check("cancelling a retry never delivers the previous failed take",
      "cancelled" in _cancelled_reply.lower() and not _cancel_state.get("video_path"), _cancelled_reply)
_E["generate_video"] = fake_gen
_E["static_continuation"] = lambda ctx, d: False
_E["revoice"] = lambda ctx, clip, voice: None

# Several faces and a spoken line: the clip's OWN speaker says it (owner,
# 2026-11-03: the silent big man got the line while the small upper man was
# the one talking in the clip).
import identity_metrics as _IDM
_real_fb = _IDM.face_boxes
_IDM.face_boxes = lambda path, pad=0.0, **k: [(10, 10, 60, 70), (100, 5, 120, 30)]
_E["clip_speaker"] = lambda ctx, src, tail="": "the smaller man at the upper left, not the large man in front"
SEEN.clear()
H._handle_generate_video(TCtx(), {}, {"description": 'He says "не ходите в театр на пролетарской" to the camera'})
check("multi-face + a spoken line: the clip's own speaker is named in the prompt",
      "the smaller man at the upper left" in SEEN.get("description", ""), SEEN.get("description", "")[:200])
check("...and the others are told to stay silent",
      "stays silent" in SEEN.get("description", ""), SEEN.get("description", "")[:250])
_IDM.face_boxes = lambda path, pad=0.0, **k: [(10, 10, 60, 70)]
SEEN.clear()
H._handle_generate_video(TCtx(), {}, {"description": 'He says "не ходите в театр" to the camera'})
check("one face: no speaker clause",
      "upper left" not in SEEN.get("description", ""), SEEN.get("description", "")[:150])
_IDM.face_boxes = _real_fb
_E["clip_speaker"] = lambda ctx, src, tail="": ""

class NewVideoCtx(TCtx):
    def __init__(self):
        super().__init__()
        self.continue_tail = self.continue_src = ""

from prompt_guard import wrap_quoted
_continuation_reads = []
class AttachedContinuationCtx(NewVideoCtx):
    last_video_path = attached_video_path = CLIP
    def yes_no(self, question, text, default=False, **kwargs):
        if 'continue or extend the video' in question:
            _continuation_reads.append(text)
            return text == 'Продолжи это видео'
        return default

_own_request = 'Продолжи это видео'
_with_source = _own_request + '\n' + wrap_quoted('speech in the attached video', 'source speech')
SEEN.clear()
H._handle_generate_video(AttachedContinuationCtx(), {'user_input_original': _with_source},
                         {'description': 'The person keeps speaking'})
check('continuation classifier sees the user request without quoted source material',
      _continuation_reads == [_own_request], _continuation_reads)
check('continuation from an attached clip uses its extracted tail', bool(SEEN.get('videos')), SEEN)

_new_state = {}
_new_reply = H._handle_generate_video(NewVideoCtx(), _new_state, {"description": "A cat walks across the room"})
check("a new video without a continuation reaches generation", bool(_new_state.get("video_path")), _new_reply)

_cancel_calls = []
_E["generate_video"] = lambda *a, **kw: (_cancel_calls.append(1) or {"status": "cancelled"})
_cancel_reply = H._handle_generate_video(NewVideoCtx(), {}, {"description": "A cat walks across the room"})
check("a cancelled render is not started again", len(_cancel_calls) == 1 and "cancelled" in _cancel_reply.lower(), _cancel_calls)
_E["generate_video"] = fake_gen

# ── the flow's own prefix: the reply says «продолжил видео», not «оживив фото» ──
import graph_personality as GP
import tg_tasks as TK
import inspect as _insp
check("the continuation prefix routes to generate_video",
      '("continue this video:", "generate_video")' in _insp.getsource(GP))
check("the continuation tail attaches for the new prefix (and the legacy one)",
      '("animate this photo", "continue this video")):' in _insp.getsource(TK))
check("the user's own words are recovered from the new prefix",
      '"continue this video: "' in _insp.getsource(TK))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
