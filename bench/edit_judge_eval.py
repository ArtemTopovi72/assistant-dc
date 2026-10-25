"""Edit-verdict eval: 59 labelled BEFORE/AFTER pairs (bench/edit_judge_set/cases.json; done 27 /
not 24 / partial 8, labelled by eye 2026-10-21 from the debug chat's edits + 6 «nothing changed»).
Run it before and after ANY change to how an edit is judged; report both scores.

    venv/Scripts/python.exe bench/edit_judge_eval.py reader       # inspect_image + the model's reading of it
    venv/Scripts/python.exe bench/edit_judge_eval.py judge        # edit_verdict.judge (one model read)
2026-10-25: the label word lists over inspect_image (gone) scored 28/63, the reading alone 26/63;
an edit with a BEFORE is judged by edit_verdict. Needs LM Studio up. Writes tmp/edit_judge_<mode>.jsonl.
"""
import json, os, sys, threading, time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for d in ("", "agent", "imaging", "media", "bot", "core", "voice"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from models import Context

CASES = json.load(open(ROOT / "bench" / "edit_judge_set" / "cases.json", encoding="utf-8"))


def _wait_for_free_card():
    """A render unloads the chat model; a call from here would load it back beside the render
    (2026-10-21: a 0.04 MP tile sat 6 min on 3.4 GB free). Wait while ComfyUI works."""
    import requests, config
    while True:
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


_SEEN: dict = {}       # case id -> the inspector's text, so both readers read the same words


def _inspect(c):
    import tool_image_handlers as H
    if c["id"] not in _SEEN:
        _SEEN[c["id"]] = H._handle_inspect_image(_ctx(), {"image_path": c["after"], "edit_before": c["before"]},
                                                 {"check": f"Is this request done on the AFTER picture: {c['ask']}?"})
    return _SEEN[c["id"]]


def reader(c):
    """The inspect_image path with the model's reading alone (no label regexes)."""
    import graph_personality as GP
    removal = c["ask"].lower().startswith(("убери", "сбрей")) and c["ask"].split(" ", 1)[-1] or ""
    res = _inspect(c)
    neg = GP._inspection_misses_request(c["ask"], res, {"inpaint_image"}, ctx=_ctx(), removal=removal)
    return {True: "not", "partial": "partial"}.get(neg, "done"), res[:400]


def judge(c):
    import edit_verdict
    v = edit_verdict.judge(_ctx(), c["before"], c["after"], c["ask"])
    return v["verdict"], v.get("why", "")


if __name__ == "__main__":
    only = set(sys.argv[2:])
    for mode in (sys.argv[1] if len(sys.argv) > 1 else "judge").split(","):
        fn = {"reader": reader, "judge": judge}[mode]
        out = ROOT / "tmp" / f"edit_judge_{mode}.jsonl"
        rows, conf = [], Counter()
        with open(out, "w", encoding="utf-8") as fh:
            for c in CASES:
                if only and c["id"] not in only:
                    continue
                _wait_for_free_card()
                t = time.time()
                try:
                    got, note = fn(c)
                except Exception as exc:
                    got, note = "error", repr(exc)[:200]
                ok = got == c["label"]
                conf[(c["label"], got)] += 1
                row = {"id": c["id"], "label": c["label"], "got": got, "ok": ok, "sec": round(time.time() - t), "note": note}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                rows.append(row)
                print(("OK  " if ok else "BAD ") + f"{c['id']:<13} want {c['label']:<8} got {got:<8} {note[:110]!r}", flush=True)
        n = len(rows); good = sum(r["ok"] for r in rows)
        # the costly mistakes: a done edit called failed (the owner's complaint) and a failed one called done
        false_fail = sum(1 for r in rows if r["label"] == "done" and r["got"] in ("not", "partial"))
        false_done = sum(1 for r in rows if r["label"] == "not" and r["got"] == "done")
        print(f"\n{mode}: {good}/{n} exact  |  done called failed: {false_fail}  |  failed called done: {false_done}")
        print("confusion (label -> got):", dict(conf))
