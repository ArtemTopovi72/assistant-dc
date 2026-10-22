"""Does an edit that puts something new on the picture take the whole-frame add path?

    venv/Scripts/python.exe bench/additive_route_eval.py old   # keyword pre-gate, then the yes/no
    venv/Scripts/python.exe bench/additive_route_eval.py new   # tool_image_handlers._adds_object
Cases: bench/additive_set/cases.json -- (region, instructions) the agent sent to inpaint_image,
30 from the logs and 5 constructed (hat, earrings, cup, balloon, moustache). Needs LM Studio.
"""
import json, os, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.join(ROOT, "bench")]
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["INTENT_LIVE"] = "1"
import requests, config
from models import Context

OLD_Q = ("An image edit names a part of the picture and what to do there:\n{text}\n\n"
         "Does it ask to ADD a new object while that part stays as it is (not to "
         "turn that part itself into something else, recolour or replace it)?")


def _idle_gpu():
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
    import image as _img, intent
    if _img.classify_edit_intent(c["instructions"], ctx=_ctx()) != "object_insert":
        return False
    return intent.ask_yes(OLD_Q, f"part: {c['region']}\nwhat to do: {c['instructions']}", default=True)


def new(c):
    import tool_image_handlers as H
    return H._adds_object(_ctx(), c["region"], c["instructions"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "new"
    cases = json.load(open("bench/additive_set/cases.json", encoding="utf-8"))
    ok = fa = fm = 0
    for c in cases:
        _idle_gpu()
        got = bool(old(c) if mode == "old" else new(c))
        good = got == c["add"]
        ok += good; fa += got and not c["add"]; fm += c["add"] and not got
        print(f"{'OK ' if good else 'BAD'} add={c['add']!s:5} got={got!s:5} | {c['region'][:30]:30} | {c['instructions'][:60]}", flush=True)
    print(f"\n{mode}: {ok}/{len(cases)} right  |  called an addition wrongly: {fa}  |  missed an addition: {fm}")
