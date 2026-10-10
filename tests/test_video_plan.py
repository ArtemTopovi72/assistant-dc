"""A multi-part clip shows its plan before any rendering starts.

User, 2026-10-10, during a 5-part kettle clip: «показать сколько частей
планируется и сколько займёт и дать что-то уплотнить, что-то выкинуть». Each
part is minutes of GPU; the chat now gets the parts, the clip length and the
render time first, and the clip renders only the parts it approves.
"""
import logging
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("F5_TEST_RUN", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T  # noqa: E402

T.redirect_data_dir(tempfile.mkdtemp(prefix="vplan_"))
import video  # noqa: E402
import tool_image_handlers as H  # noqa: E402

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond)
    BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")


SCRIPT = ("Мужчина берёт чайник, ставит его на плиту и ждёт несколько секунд. Затем чайник "
          "начинает свистеть, мужчина вздрагивает и говорит «Ну наконец-то, сколько можно ждать». "
          "Потом он наливает чай, долго дует на чашку несколько секунд, пробует и морщится. "
          "После этого он смотрит в камеру и говорит «Горячо, как в аду, но вкусно, честное слово». "
          "Затем он ставит чашку, вытирает руки полотенцем и уходит из кадра.")

# ── the tool stops on a plan instead of rendering ─────────────────────────────
H.asks_for_video = lambda ctx, state: True
video.engine_available = lambda ctx: (True, "")
import tools as _t  # noqa: E402
_t._render_budget_exhausted = lambda *a, **k: None
rendered = []
video.generate_video = lambda *a, **k: rendered.append(a) or {}
ctx = SimpleNamespace(video_plan_ask=True, video_plan_parts=[], set_stage=lambda s: None,
                      continue_tail="", anim_voices=[], voice_choice="default")
state = {}
out = H._handle_generate_video(ctx, state, {"description": SCRIPT, "use_current_images": False})
plan = state.get("video_plan") or []
check("a script of several parts stops before rendering", out.startswith("[NOT MADE YET]") and not rendered, out[:80])
check("...and hands over every part with its seconds", len(plan) > 1 and all(p["sec"] > 0 for p in plan), plan)

# ── the card in the chat ──────────────────────────────────────────────────────
bot = T.TelegramBot("123:TEST", lambda: None, lambda: object(), lambda: {"messages": []}, silent_mode=True)
bot._backend = T.InMemoryBackend()
sent, queued = [], []
bot._send_text = lambda cid, t, **k: (sent.append((t, k.get("keyboard"))), 1)[1]
bot._enqueue_item = lambda cid, item: queued.append(item)
bot._activity.log = lambda *a, **k: None
CID = 9_300_888
bot._user_store.put(T._User(chat_id=CID, name="V", status="approved", is_admin=False))
sess = bot._get_session(CID)
sess.lang = "ru"
bot._store.put(sess)

bot._offer_video_plan(CID, sess, "ru", "сними видео: " + SCRIPT, plan)
text, kb = sent[-1]
datas = [b["callback_data"] for row in kb["inline_keyboard"] for b in row]
check("the card names the parts, the clip length and the render time",
      f"частей {len(plan)}" in text and "ролик ≈" in text and "рендер ≈" in text, text[:200])
check("...with render, condense, drop-a-part and cancel buttons",
      {"vp:go", "vp:squeeze", "vp:x", "vp:del:0"} <= set(datas), datas)

bot._cb_video_plan(CID, "vp:del:0")
sess = bot._get_session(CID)
check("🗑 drops that part", len(sess.video_plan) == len(plan) - 1, len(sess.video_plan))

bot._cb_video_plan(CID, "vp:go")
sess = bot._get_session(CID)
check("▶ reruns the same request", queued and queued[-1]["text"] == "сними видео: " + SCRIPT, queued)
check("...with the approved parts saved for that run", sess.video_plan_ok == [p["text"] for p in plan[1:]],
      sess.video_plan_ok)

# ── the rerun renders exactly the approved parts ─────────────────────────────
ctx2 = SimpleNamespace(video_plan_ask=True, video_plan_parts=list(sess.video_plan_ok),
                       set_stage=lambda s: None, continue_tail="", anim_voices=[], voice_choice="default")
state2 = {}
try:
    H._handle_generate_video(ctx2, state2, {"description": SCRIPT, "use_current_images": False})
except Exception:
    pass
check("the approved run does not ask again", "video_plan" not in state2, state2.get("video_plan"))
check("...and its first render is the first approved part", rendered and sess.video_plan_ok[0] in str(rendered[0][1]),
      rendered[:1])

print(f"\n{OK}/{OK + BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
