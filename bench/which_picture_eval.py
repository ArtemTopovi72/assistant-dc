"""«Какую картинку?» -- does the bot ask exactly when the chat leaves open which picture?

    venv/Scripts/python.exe bench/which_picture_eval.py old   # the word gate the bot shipped with
    venv/Scripts/python.exe bench/which_picture_eval.py new   # tg_bot._message_subject
Cases: bench/which_picture_set/cases.json -- the chat lines before each message (chatlog.lines_of)
and what it was about; 24 from real chats, 5 constructed where asking is right. Needs LM Studio.
"""
import json, os, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# bench/ holds a knowledge.py that shadows knowledge/; the project dirs come from the venv's .pth
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.join(ROOT, "bench")]
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import requests, config
from models import Context


def _idle_gpu():
    """A render unloads the chat model; a read from here would load it back beside the render."""
    for _ in range(360):
        try:
            q = requests.get(config.COMFY_URL + "/queue", timeout=5).json()
            if not q.get("queue_running") and not q.get("queue_pending"):
                return
        except Exception:
            return
        time.sleep(10)


def _ctx():
    c = Context(models=None, transcription_cache={}, cache_file=None, asr_lock=threading.Lock(),
                tts_lock=threading.Lock())
    c.set_stage = lambda *a, **k: None
    return c


def old(c):
    import tg_bot
    prev = next((r[5:] for r in reversed(c["recent"]) if r.startswith("bot: ")), "")
    return "ask" if tg_bot._needs_image_choice(c["text"], ctx=_ctx(), previous=prev) else "no"


def new(c):
    import tg_bot
    s = tg_bot._message_subject(c["text"], c["recent"], ctx=_ctx())
    return ("ask" if s == "unclear" else "no"), s


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "new"
    cases = json.load(open("bench/which_picture_set/cases.json", encoding="utf-8"))
    ok = wrong_ask = missed_ask = subj_ok = 0
    for c in cases:
        _idle_gpu()
        got = old(c) if mode == "old" else new(c)
        ask, subj = (got, "") if isinstance(got, str) else got
        want_ask = "ask" if c["want"] == "unclear" else "no"
        good = ask == want_ask
        ok += good; subj_ok += subj == c["want"]
        wrong_ask += ask == "ask" and want_ask == "no"; missed_ask += ask == "no" and want_ask == "ask"
        print(f"{'OK ' if good else 'BAD'} {c['id'][:28]:28} want {c['want']:9} got {ask:3} {subj:9} | {c['text'][:50]!r}", flush=True)
    print(f"\n{mode}: {ok}/{len(cases)} right  |  asked wrongly: {wrong_ask}  |  should have asked: {missed_ask}"
          + (f"  |  subject exact: {subj_ok}/{len(cases)}" if mode == "new" else ""))
