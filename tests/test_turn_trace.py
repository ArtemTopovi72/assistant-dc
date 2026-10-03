"""turn_trace: every turn leaves one line with phases, LLM calls, log events and
the draft a guard replaced. Run: venv/Scripts/python.exe tests/test_turn_trace.py
"""
import json
import logging
import os
import sys
import tempfile
import threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "agent"), os.path.join(ROOT, "core")]
os.environ["TURNS_DIR"] = tempfile.mkdtemp()
import turn_trace as TT  # noqa: E402

bad = 0


def check(name, cond, extra=""):
    global bad
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   {extra}")); bad += not cond


log = logging.getLogger("assistant.graph"); log.setLevel(logging.INFO)
tok = TT.start(chat=1, task="t1", text="что из этого взять?")
TT.stage("Thinking"); TT.llm("You are a helpful", {"prompt_tokens": 900, "completion_tokens": 120}, 2.5, "stop", 400)
log.info("Round 0: forcing calculate")
log.warning("Answer says a file is ready but none exists — replacing")
TT.stage("Speaking")
TT.answer("draft", "Борис говорит, презентация готова"); TT.answer("final", "Готового файла у меня нет")
# another thread's log lines are not this turn's
th = threading.Thread(target=lambda: log.warning("other chat's warning")); th.start(); th.join()
path = TT.finish(tok, outcome="done")
check("a trace file is written", path and os.path.exists(path), path)
t = json.loads(open(path, encoding="utf-8").read().splitlines()[-1])
check("phases carry durations", [p["name"] for p in t["phases"]] == ["Thinking", "Speaking"]
      and t["phases"][0]["s"] is not None, t["phases"])
check("llm call with tokens", t["totals"]["tokens_in"] == 900 and t["totals"]["tokens_out"] == 120, t["totals"])
msgs = [e["msg"] for e in t["events"]]
check("guard and forcing lines captured", any("replacing" in m for m in msgs) and any("forcing" in m for m in msgs), msgs)
check("other threads' lines are not mixed in", not any("other chat" in m for m in msgs), msgs)
check("the replaced draft is kept", "презентация готова" in t["answers"]["draft"], t["answers"])
check("no trace open after finish", TT.current() is None)
log.warning("after the turn")
tok = TT.start(chat=2)
TT.answer("draft", "Курс **83.5588** рубля."); TT.answer("final", "Курс 83,5588 рубля")
t2 = json.loads(open(TT.finish(tok), encoding="utf-8").read().splitlines()[-1])
check("a reformatted answer is not a replaced draft", "draft" not in t2["answers"], t2["answers"])
check("lines after the turn go nowhere", "after the turn" not in json.dumps(t, ensure_ascii=False))


def last(path):
    return json.loads(open(path, encoding="utf-8").read().splitlines()[-1])


# Work handed to another thread is part of the turn (live 16:11: a forwarded
# video's retelling ran on its own thread and left no trace at all).
day = os.path.join(os.environ["TURNS_DIR"], os.listdir(os.environ["TURNS_DIR"])[0])
n_before = len(open(day, encoding="utf-8").read().splitlines())
go = threading.Event()


def retell():
    go.wait(5)
    TT.llm("You retell forwarded VIDEO messages", {"prompt_tokens": 700, "completion_tokens": 90}, 1.2, "stop")
    log.warning("retell guard fired")
    TT.answer("final", "Основная тема: телевизор")


tok = TT.start(chat=3, task="button", text="fwdv:sum")
th = TT.spawn(retell, name="fwd-retell")
check("a turn with a thread still running is not written yet", TT.finish(tok, keep_idle=False) == "")
go.set(); th.join(5)
n_after = len(open(day, encoding="utf-8").read().splitlines())
t3 = last(day)
check("the last thread writes it, with the thread's model call, log line and answer",
      n_after == n_before + 1 and t3["meta"]["chat"] == 3 and t3["totals"]["tokens_in"] == 700
      and any("retell guard" in e["msg"] for e in t3["events"]) and t3["answers"].get("final"), t3)
check("bookkeeping fields are not written", not any(k.startswith("_") for k in t3), list(t3))

tok = TT.start(chat=4, task="search")
with TT.Pool(4) as ex:
    list(ex.map(lambda i: TT.llm(f"read page {i}", {"prompt_tokens": 10, "completion_tokens": 1}, 0.1), range(4)))
t4 = last(TT.finish(tok))
check("a pool's parallel calls belong to the turn that submitted them", t4["totals"]["llm_calls"] == 4, t4["totals"])

tok = TT.start(chat=5, task="button", text="nav:close")
check("a press that reached no model and logged nothing is not written",
      TT.finish(tok, keep_idle=False) == "" and last(day)["meta"]["chat"] == 4)

# The whole call is kept: what was sent (pictures as files), what came back,
# and the file the user sent. Live 2026-10-01: a projector was «красный
# цилиндр» and the trace held only each prompt's first 80 characters.
import base64
tok = TT.start(chat=6, task="batch")
TT.media_bytes(b"VIDEO-BYTES", ".mp4")
img = base64.b64encode(b"JPEG-BYTES").decode()
TT.llm("You see one frame", {}, 0.5, "stop", 11,
       messages=[{"role": "system", "content": "You see one frame"},
                 {"role": "user", "content": [{"type": "text", "text": "what is it?"},
                  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]}],
       reply={"content": "a red cylinder", "tool_calls": None})
t6 = last(TT.finish(tok))
c = t6["llm"][0]
pic = c["messages"][1]["content"][1].get("image", "")
check("the reply text is kept", c["reply"] == {"content": "a red cylinder"}, c.get("reply"))
check("the prompt is kept whole", c["messages"][1]["content"][0]["text"] == "what is it?", c["messages"])
check("the picture sent is a file, not base64", pic and open(pic, "rb").read() == b"JPEG-BYTES", pic)
check("the user's file is kept with the turn",
      open(t6["media"][0]["path"], "rb").read() == b"VIDEO-BYTES", t6.get("media"))
sys.exit(1 if bad else 0)
