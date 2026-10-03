"""«как меня зовут и что ты обо мне знаешь?» forgot Марат three turns later
(journey 9, live 2026-09-18).

_scoped_ctx() hands every Telegram task a FRESH, empty pinned_facts list (by
design -- one chat must never see another's facts mid-turn). remember_fact
saves the fact to tg_memory/chat_<id>/facts.json immediately, but nothing
ever read that file back into the NEXT task's ctx: the "Saved facts" block
that graph_compose.py injects every turn was always empty in production,
and a fact only "worked" for as long as the raw exchange survived in the
chat history -- until the 3-turn rolling compaction folded it away. Now
_execute_task calls ctx.load_memory(_mem_dir) right after pointing
active_memory_dir at the per-chat folder, so a fact saved in an earlier
task is visible again in this one.
"""
import os, sys, tempfile, threading, types
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
import models as M

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_factspersist_")
T.redirect_data_dir(_DATA_DIR)

CID = 998833
_shared = M.Context(
    models=types.SimpleNamespace(), transcription_cache={},
    cache_file=T._MEMORY_DIR / "cache.json",
    asr_lock=threading.Lock(), tts_lock=threading.Lock(),
)
_mem_dir = T._MEMORY_DIR / f"chat_{CID}"

# --- task 1: the fact is saved -----------------------------------------------------
ctx1 = T._scoped_ctx(_shared)
_mem_dir.mkdir(parents=True, exist_ok=True)
ctx1.active_memory_dir = _mem_dir
ctx1.load_memory(_mem_dir)          # nothing saved yet
check("a brand-new chat starts with no saved facts", ctx1.facts_text() == "")
added = ctx1.pin_fact("Пользователя зовут Марат, он вегетарианец и у него аллергия на орехи.")
ctx1.save_memory(_mem_dir)
check("remember_fact saved the fact", added and "Марат" in ctx1.facts_text())

# --- task 2: a FRESH scoped ctx, three turns later (same shape as _execute_task) ---
ctx2 = T._scoped_ctx(_shared)
check("a new task starts with an empty pinned_facts list (per-chat isolation)",
      ctx2.pinned_facts == [])
ctx2.active_memory_dir = _mem_dir
ctx2.load_memory(_mem_dir)          # the fix: read back what task 1 saved
check("the fact survives into the next task", "Марат" in ctx2.facts_text())
check("...allergy detail too", "орех" in ctx2.facts_text())

# --- the wiring is actually in _execute_task, not just possible in principle ------
src = open("bot/tg_tasks.py", encoding="utf-8").read()
i_dir = src.index("ctx.active_memory_dir = _mem_dir")
i_load = src.index("ctx.load_memory(_mem_dir)")
check("_execute_task loads memory right after pointing at the per-chat dir",
      0 < i_dir < i_load < i_dir + 600)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
