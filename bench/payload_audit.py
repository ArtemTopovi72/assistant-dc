"""How big is what we actually send, per kind of request -- and what is in it.

Drives the REAL agent loop with a model stub that records the first payload of
each turn and answers at once, so it measures exactly what production sends:
system prompt, pinned facts, history and the tool schemas retrieval picked.

    venv/Scripts/python bench/payload_audit.py
    venv/Scripts/python bench/payload_audit.py --sandbox      # with the file kit

Tokens are estimated at 3.3 chars/token (mixed RU/EN JSON); use it to compare
runs, not as an exact count. A tool listed under a request it has nothing to
do with is a retrieval bug worth fixing: every extra schema is a distraction
and ~200-600 tokens.
"""
import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import logging
logging.basicConfig(level=logging.CRITICAL)

import graph as graph_mod
import llm as llm_mod
from config import MODEL_NAME
from models import Context
from prompts import SYSTEM_PROMPT_PERSONALITY

CHARS_PER_TOKEN = 3.3

REQUESTS = [
    ("chat", "привет, как дела?"),
    ("fact", "сколько будет 17 * 23 - 91?"),
    ("weather", "какая погода в Казани завтра?"),
    ("search", "найди в интернете, когда выходит новая версия Blender"),
    ("research", "сделай глубокое исследование рынка электросамокатов в России"),
    ("draw", "нарисуй кота в космосе"),
    ("edit_img", "убери с фото человека на заднем плане"),
    ("ozon", "найди на озоне самый дешёвый электрочайник"),
    ("basket", "собери на озоне набор для пикника"),
    ("remember", "запомни, что я живу в Казани"),
    ("deck", "сделай презентацию про солнечную систему на 5 слайдов"),
    ("code", "почини ошибку в script.py, он падает"),
    ("mod", "тут два джарника, добавь блоки из одного в тег другого"),
    ("docs", "что сказано в моих документах про отпуск?"),
]


class _User:
    prefs = {"sandbox": "code"}


def _ctx(tmp: Path, sandbox: bool):
    c = Context(models=None, transcription_cache={}, cache_file=tmp / "c.json",
                asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                model_name=MODEL_NAME, no_think=True)
    c.tts_disabled = True
    c.gui_mode = True
    c.web_search_enabled = True
    c.active_memory_dir = tmp / "mem"
    c.active_memory_dir.mkdir(exist_ok=True)
    if sandbox:
        import sandbox_access as A
        from code_sandbox import Sandbox
        box = Sandbox(tmp / "sbx")
        box.write_text("script.py", "print(1/0)\n")
        c.sandbox, c.sandbox_user = box, _User()
        _User.prefs = {"sandbox": A.CODE}
    return c


def measure(text: str, sandbox: bool) -> dict:
    seen = {}

    def send(ctx, messages, tools=None, **k):
        if "messages" not in seen and tools:
            seen["messages"] = [dict(m) for m in messages]
            seen["tools"] = list(tools or [])
        return {"role": "assistant", "content": "ok"}

    graph_mod.send_to_lm_studio = send
    llm_mod.send_to_lm_studio = send
    with tempfile.TemporaryDirectory() as d:
        c = _ctx(Path(d), sandbox)
        st = {"messages": [{"role": "system", "content": SYSTEM_PROMPT_PERSONALITY}],
              "user_input": text, "image_data": None, "final_answer": "",
              "session_memory_text": "", "vision_summary": "", "image_path": "",
              "image_score": 0, "image_attempt": 0, "image_status": ""}
        saved = graph_mod._TOOL_TRIGGER_RE
        graph_mod._TOOL_TRIGGER_RE = __import__("re").compile("")
        try:
            graph_mod.build_graph(c).invoke(st)
        finally:
            graph_mod._TOOL_TRIGGER_RE = saved
    msgs, tools = seen.get("messages") or [], seen.get("tools") or []
    sys_chars = sum(len(str(m.get("content") or "")) for m in msgs if m.get("role") == "system")
    other = sum(len(str(m.get("content") or "")) for m in msgs if m.get("role") != "system")
    tool_chars = {((t.get("function") or {}).get("name")): len(json.dumps(t, ensure_ascii=False))
                  for t in tools}
    return {"system": sys_chars, "other": other, "tools": tool_chars}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--json", help="also write the rows here")
    a = ap.parse_args(argv)
    rows = []
    tok = lambda n: int(n / CHARS_PER_TOKEN)
    print(f"{'kind':10} {'total':>7} {'system':>7} {'tools':>7} {'#':>3}  tools sent")
    for kind, text in REQUESTS:
        r = measure(text, a.sandbox)
        tools_total = sum(r["tools"].values())
        total = r["system"] + r["other"] + tools_total
        rows.append({"kind": kind, "text": text, "total_tokens": tok(total),
                     "system_tokens": tok(r["system"]), "tool_tokens": tok(tools_total),
                     "tools": sorted(r["tools"])})
        print(f"{kind:10} {tok(total):7d} {tok(r['system']):7d} {tok(tools_total):7d} "
              f"{len(r['tools']):3d}  {', '.join(sorted(r['tools']))}")
    biggest = {}
    for kind, text in REQUESTS[:1]:
        biggest = measure(text, a.sandbox)["tools"]
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nlargest schemas: " + ", ".join(
        f"{k} {tok(v)}" for k, v in sorted(biggest.items(), key=lambda kv: -kv[1])[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
