"""Live 2026-09-29: menus piled up, answers were not threaded to the request, every
answer came twice (voice + the same text), a button's English payload got an
English reply in a Russian chat, and the OCR worker's cp1251 pipe broke on «ы»."""
import os, sys, tempfile, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import tg_bot as T, tg_transport, tg_sessions, graph_language as GL

# reply mode: text by default, cycles, old voice_on setter still works
s = tg_sessions._Session(1)
assert s.reply_mode == "text" and not s.voice_on
s.voice_on = True; assert s.reply_mode == "both"
assert "reply_mode" in s.to_dict() and tg_sessions._Session(1, s.to_dict()).reply_mode == "both"
print("PASS reply mode")

# every send inside a task is a reply to the request
sent = []
tg_transport.requests.post = lambda url, **kw: sent.append((url, kw)) or type("R", (), {"json": lambda self: {}, "status_code": 200})()
with tg_transport.replying_to(5, 77):
    tg_transport._logged_post("x/sendMessage", json={"chat_id": 5, "text": "hi"})
    tg_transport._logged_post("x/sendVoice", data={"chat_id": 5}, files={})
    tg_transport._logged_post("x/sendMessage", json={"chat_id": 6, "text": "other chat"})
tg_transport._logged_post("x/sendMessage", json={"chat_id": 5, "text": "after"})
rp = [kw.get("json", kw.get("data")).get("reply_parameters") for _, kw in sent]
assert rp[0]["message_id"] == 77 and json.loads(rp[1])["message_id"] == 77 and rp[2] is None and rp[3] is None, rp
print("PASS replies threaded")

# menu hopping wipes the previous press and its menu reply
T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_ux_"))
bot = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
bot._user_store.put(T._User(chat_id=9, name="F", status="approved"))
deleted, n = [], [100]
bot._delete = lambda c, m: deleted.append(m)
def api(method, p=None, **k):
    n[0] += 1; return {"ok": True, "result": {"message_id": n[0]}}
bot._api_post = api
bot._enqueue_item = lambda cid, item: bot._send_text(cid, "⚙️ Настройки", keyboard={"keyboard": [["x"]]})
lab = T._b("settings", "ru")
for mid in (1, 2):
    bot._dispatch_logged({"message": {"message_id": mid, "chat": {"id": 9, "type": "private"},
                                      "from": {"id": 9}, "date": int(time.time()), "text": lab}})
assert 1 in deleted and 101 in deleted and 2 not in deleted, deleted
print("PASS menu tail")

# a button payload in a Russian chat gets its English reply translated
class C:
    reply_lang = "ru"
    def is_cancelled(self): return False
GL.call_llm_simple = lambda ctx, sys_, txt, **k: "Не получилось убрать надпись полностью."
out = GL._match_reply_language(C(), "Failed to completely remove the text from the image, sorry.",
                               "remove all the lettering from the image")
print("lang:", out)
assert out and "Не получилось" in out, out
print("PASS ru payload")

import inspect, ocr_worker
assert "ensure_ascii=False" not in inspect.getsource(ocr_worker)
print("PASS")

import ideogram_layout as L
_d = L.normalize_layout({"elements": [{"desc": "a smooth-faced cat with no whiskers and a clean muzzle, looking up", "x": 0, "y": 0, "w": .5, "h": .5},
                                      {"desc": "a No Parking sign", "text": "No Parking", "x": .5, "y": 0, "w": .3, "h": .2}]})["elements"]
assert _d[0]["desc"] == "a smooth-faced cat and a clean muzzle, looking up" and _d[1]["desc"] == "a No Parking sign", _d
print("PASS absent things are not named to the renderer")

# someone else's forwarded question is not a question about the user's last photo
import graph as G, intent
intent.STUB = {"а это кто?": {"is_question": True}}.get   # the model's read
class _Cx: last_image_path = __file__; image_pointed_at = False
from prompt_guard import wrap_quoted as _wq
_fwd = (_wq("the user forwarded this message from someone else; it is not addressed to you",
            "За кого проходит?") + "\nНе, в деребас уехал")
_orig_stem = G._USER_UPLOAD_STEM
G._USER_UPLOAD_STEM = os.path.basename(__file__)[:5]
try:
    assert not G.needs_relook(_Cx(), {"user_input": "Who is he playing as?", "user_input_original": _fwd})
    assert G.needs_relook(_Cx(), {"user_input": "Who is he?", "user_input_original": "а это кто?"})
finally:
    G._USER_UPLOAD_STEM = _orig_stem
print("PASS forwarded questions do not re-read the photo")

# a restated fact replaces the old one; a look-alike stays
import threading, tools as TL, llm as _llm
class _FC:
    def __init__(self): self.pinned_facts, self.memory_lock = [], threading.Lock()
    def pin_fact(self, t): self.pinned_facts.append({"text": t}); return True
    def save_memory(self, d): pass
    active_memory_dir = ""
_asked = []
_llm.call_llm_simple = lambda ctx, p, body, **k: _asked.append(body) or ("1" if "NEW: Собаку пользователя зовут Бобик" in body else "none")
c = _FC()
for f in ("Собаку пользователя зовут Жужа", "Любимый цвет пользователя — бирюзовый"):
    TL._handle_remember_fact(c, {}, {"fact": f})
r = TL._handle_remember_fact(c, {}, {"fact": "Собаку пользователя зовут Бобик"})
assert [f["text"] for f in c.pinned_facts] == ["Любимый цвет пользователя — бирюзовый", "Собаку пользователя зовут Бобик"], c.pinned_facts
assert "Жужа" in r
TL._handle_remember_fact(c, {}, {"fact": "Кошку пользователя зовут Мурка"})
assert len(c.pinned_facts) == 3, c.pinned_facts
print("PASS a restated fact replaces the old one")

import context_v2 as CV
_st = CV.empty_state()
CV.merge_delta(_st, {"add_decisions": ["User: нарисуй кота", "Кот нарисован в стиле аниме",
                                       "<|tool_call>call:generate_image{prompt:cat}"]})
assert _st["decisions"] == ["Кот нарисован в стиле аниме"], _st["decisions"]
print("PASS transcript-shaped memory items are dropped")
