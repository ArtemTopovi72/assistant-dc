"""Golden task set: does the REAL agent finish common tasks end to end -- picks the tool,
fills it right, uses what the tool returned, answers -- and what does a turn cost.

Live LM Studio and the real loop; every tool is answered with a canned result instead of
run (no render, no web). A case passes when the tool is one of the expected ones, its
arguments carry the expected words, and the reply carries the fact the tool returned.
Measure before and after every prompt/routing change, one change at a time.

    venv/Scripts/python.exe bench/golden_eval.py [--trials 2] [--only weather,calc] [--out tmp/golden/x.json]
"""
import argparse
import json
import os
import sys
import threading
import types
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "bench"]   # bench/knowledge.py shadows the package
for d in ("", "agent", "imaging", "media", "bot", "core", "voice", "knowledge", "research", "services", "sandbox", "personalities"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT)
os.environ.setdefault("TURNS_DIR", str(ROOT / "tmp" / "golden" / "turns"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger().handlers[0].setLevel(logging.WARNING)
logging.getLogger("assistant").setLevel(logging.INFO)   # the turn trace keeps INFO events

import graph
import turn_trace
from app_runtime import make_base_state
from models import Context

PICTURE = str(ROOT / "bench" / "assets" / "red_carpet.jpg")
EDITS = ("inpaint_image", "redraw_image", "generate_image", "transfer_image", "fix_hands", "fix_artifact", "swap_person")

# name, request, expected tools (() = no tool), words one of which each arg group must hold,
# canned result, words one of which the reply must hold, picture attached
CASES = [
    ("weather", "какая погода завтра в Казани?", ("weather_forecast",), [("казан", "kazan")],
     "Kazan, tomorrow: +7°C, light rain, wind 5 m/s", ("7",), False),
    ("calc", "сколько будет 1234 умножить на 5678?", ("calculate",), [("1234",)],
     "7006652", ("7006652", "7 006 652", "7 006 652"), False),
    ("draw", "нарисуй кота-космонавта на Луне", ("generate_image",), [("cat", "кот"), ("moon", "лун")],
     None, (), False),
    ("search", "найди в интернете, кто выиграл чемпионат мира по шахматам 2024 года", ("search", "deep_research"),
     [("2024",)], "World Chess Championship 2024 (Singapore): Gukesh Dommaraju beat Ding Liren 7.5-6.5.",
     ("гукеш", "gukesh", "Гукеш"), False),
    ("remind", "напомни мне через 20 минут выключить духовку", ("set_reminder",), [("20",), ("духов", "oven")],
     "Reminder set for 20 minutes from now: выключить духовку", (), False),
    ("currency", "сколько будет 100 долларов в рублях?", ("convert_currency",), [("100",), ("usd", "доллар")],
     "100 USD = 8150.00 RUB (rate 81.50)", ("8150", "8 150", "8 150"), False),
    ("ozon", "найди на озоне беспроводные наушники до 3000 рублей", ("ozon_search",), [("наушник", "headphone", "earbud"), ("3000", "3 000")],
     "Found 3 items: 1) JBL Tune 520BT -- 2 790 ₽, rating 4.8; 2) Xiaomi Redmi Buds 6 -- 2 150 ₽, rating 4.7; "
     "3) Honor Choice Earbuds X5 -- 2 490 ₽, rating 4.8", ("2 150", "2150", "2 150", "redmi"), False),
    ("deck", "сделай презентацию на 5 слайдов про историю космонавтики", ("create_presentation",), [("космонавт", "space", "cosmonaut")],
     "Presentation created: 5 slides, saved to runtime/out/space_history.pptx and sent.", (), False),
    ("video", "сделай видео: собака бежит по пляжу на закате", ("generate_video",), [("dog", "собак"), ("beach", "пляж")],
     "Video generated and saved: runtime/out/dog_beach.mp4 (6.0 s)", (), False),
    ("code", "запусти код на питоне, который выведет первые 10 чисел Фибоначчи", ("run_code",), [("print",)],
     "exit 0\nstdout:\n0 1 1 2 3 5 8 13 21 34", ("34",), False),
    ("time", "который сейчас час в Токио?", ("local_time",), [("tokyo", "токио", "asia/tokyo")],
     "Tokyo (Asia/Tokyo): 2026-10-27 02:14, Tuesday", ("2:14", "02:14"), False),
    ("remember", "запомни, что у меня аллергия на арахис", ("remember_fact",), [("арахис", "peanut")],
     "Saved.", (), False),
    ("erase", "убери с фотографии всех людей на заднем плане", EDITS, [("background", "задн", "people", "люд")],
     None, (), True),
    ("hello", "привет, как дела?", (), [], None, (), False),
    ("translate", "переведи на английский: я люблю зимний лес", (), [], None, ("forest",), False),
]


def _fresh() -> None:
    """Per-message read caches off: a case read before would cost nothing the second time."""
    for mod, names in (("intent", ("_CACHE", "_YES_CACHE")), ("ask_read", ("_CACHE",)),
                       ("reply_read", ("_CACHE",)), ("image_router", ("_EDIT_CACHE", "_REGION_CACHE"))):
        m = sys.modules.get(mod)
        for n in names:
            getattr(m, n, {}).clear() if m else None


def run_case(case) -> dict:
    name, request, want_tools, arg_words, canned, reply_words, with_picture = case
    calls = []
    _fresh()

    def record(ctx, state, tool, args):
        calls.append((tool, args))
        if tool == "generate_image":   # the real handler's accepted-render shape
            state["image_path"], state["image_status"] = PICTURE, "success"
            state["fresh_render"] = {"path": PICTURE, "score": 9, "status": "success", "accepted": True}
            return (f"Image generated and saved: {PICTURE}\nScore: 9/10, attempts: 1, status: success\n"
                    "The picture has already been checked against the request by the vision model and "
                    "accepted. Do NOT inspect it again and do NOT edit it on your own: answer the user "
                    "and deliver it as it is. If they want something changed, they will say so.")
        if tool in EDITS:
            state["image_path"], state["image_status"] = PICTURE, "success"
            return f"Image edited and saved: {PICTURE}"   # the real handlers' success shape
        # the loop takes a deck or clip with no file behind it as a failure
        if tool == "create_presentation":
            state["document_path"] = str(ROOT / "runtime" / "out" / "space_history.pptx")
        if tool == "generate_video":
            state["video_path"] = str(ROOT / "runtime" / "out" / "dog_beach.mp4")
        return canned if canned is not None else f"[{tool} done]"

    ctx = Context(models=None, transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.no_think, ctx.tts_disabled, ctx.gui_mode = True, True, True
    ctx.run_tool = record
    # run_code is offered only with a sandbox the user may execute in
    ctx.sandbox = object()
    ctx.deps = ctx.deps if getattr(ctx, "deps", None) is not None else types.SimpleNamespace()
    ctx.deps.may_run_code = lambda user: True
    state = make_base_state()
    state["user_input"] = request
    if with_picture:
        ctx.last_image_path = PICTURE
    token = turn_trace.start(chat="golden", task=name, text=request)
    t0 = time.monotonic()
    err = ""
    try:
        final = graph.build_graph(ctx).invoke(state)
        reply = final.get("final_answer") or ""
    except Exception as e:
        reply, err = "", f"{type(e).__name__}: {e}"
    seconds = time.monotonic() - t0
    trace = turn_trace.current() or {}
    turn_trace.finish(token)
    llm = trace.get("llm", [])
    main_in = [c.get("in") or 0 for c in llm if str(c.get("purpose", "")).startswith("You are a helpful voice assistant")]

    tools = [c[0] for c in calls if c[0] not in ("update_plan", "fold_context", "recall_context")]
    why = []
    if want_tools:
        hit = [c for c in calls if c[0] in want_tools]
        if not hit:
            why.append(f"tool {tools or 'none'} not in {list(want_tools)}")
        else:
            blob = json.dumps(hit[0][1], ensure_ascii=False).lower()
            for group in arg_words:
                if not any(w.lower() in blob for w in group):
                    why.append(f"args lack {group}")
    elif tools:
        why.append(f"called {tools} for a chat answer")
    low = reply.lower()
    if not reply.strip():
        why.append("empty reply")
    if reply_words and not any(w.lower() in low for w in reply_words):
        why.append(f"reply lacks {reply_words}")
    if err:
        why.append(err)
    return {"case": name, "ok": not why, "why": why, "tools": tools,
            "args": calls[0][1] if calls else None, "reply": reply[:400],
            "seconds": round(seconds, 1), "llm_calls": len(llm),
            "llm_seconds": round(sum(c.get("s", 0) for c in llm), 1),
            "main_in": max(main_in) if main_in else 0,
            "purposes": [" ".join(str(c.get("purpose", "")).split())[:60] for c in llm]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    only = {s for s in a.only.split(",") if s}
    cases = [c for c in CASES if not only or c[0] in only]
    rows = []
    for trial in range(a.trials):
        for c in cases:
            r = run_case(c)
            r["trial"] = trial
            rows.append(r)
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"[{trial}] {mark} {r['case']:<10} {r['seconds']:>5}s calls={r['llm_calls']:<2} "
                  f"llm={r['llm_seconds']:>5}s main_in={r['main_in']:<6} tools={r['tools']} "
                  f"{'; '.join(r['why'])}", flush=True)
            if not r["ok"]:
                print("      reply:", r["reply"][:200].replace("\n", " | "), flush=True)
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    avg = lambda k: round(sum(r[k] for r in rows) / max(n, 1), 1)
    print(f"\nPASS {ok}/{n}  avg turn {avg('seconds')}s  llm calls {avg('llm_calls')}  "
          f"llm {avg('llm_seconds')}s  main prompt {avg('main_in')} tok")
    out = Path(a.out or ROOT / "tmp" / "golden" / f"run_{time.strftime('%m%d_%H%M')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
