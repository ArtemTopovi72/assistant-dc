"""UX audit over the real per-chat transcripts (chat_logs/*.jsonl, chatlog.py).

Unit suites check each piece; they cannot see what the USER lived through.
2026-09-27: «🎵 Songs» was 'layout-fixed' into «Ыщтпы», redrew a photo, Cancel
said «Cancelling…» and the render went on, a typed «Stop» became a steer note —
every suite was green. Each rule below is a symptom a person would notice.

Run:  venv/Scripts/python.exe scripts/ux_audit.py [--since "2026-09-27"] [DIR]
Exit 1 when any ERROR is found.
"""
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_STOP = {"stop", "стоп", "хватит", "отмена", "отмени", "cancel", "остановись"}
_WORK = re.compile(r"Got it|redrawing|Redrawing|Drawing|Рисую|Перерисов|Editing|Редактир|"
                   r"Writing a response|Пишу ответ|Starting|Начинаю|Looking at|Смотрю")
_DONE_CANCEL = re.compile(r"Request cancelled|Запрос отменён|already finished|уже завершён|"
                          r"Stopped|Остановлено|Nothing was running|Ничего не выполн")


def _labels():
    try:
        import tg_strings
        return {v for forms in tg_strings._BTN.values() for v in forms.values()}
    except Exception:
        return set()


def _ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def audit_chat(rows, labels):
    """rows: parsed jsonl records of ONE chat. Returns [(level, ts, rule, detail)]."""
    out = []
    ins = [(r, (r["data"].get("text") or "").strip()) for r in rows
           if r["kind"] == "in" and isinstance(r["data"], dict) and r["data"].get("text")]
    outs = [r for r in rows if r["kind"] == "out" and isinstance(r["data"], dict)]
    otext = lambda r: r["data"].get("text") or ""

    # 1. the layout fixer rewrote something that was a button or a known word
    for r in outs:
        m = re.match(r"⌨️.*?«(?:<i>)?(.*?)(?:</i>)?»", otext(r))
        if m:
            import keyboard_layout as kl
            back = m.group(1).translate(str.maketrans(kl._RU, kl._EN))
            lvl = "ERROR" if back in labels else "WARN"
            out.append((lvl, r["ts"], "layout-rewrite",
                        f"«{back}» was rewritten to «{m.group(1)}»"))

    # 2. the same thing sent 3+ times within 5 minutes: the user is not getting what they want
    nav = {l for l in labels if l.startswith(("↩", "🏠"))}      # Back / Main menu: navigation
    for i, (r, t) in enumerate(ins):
        if t in nav or not any(c.isalpha() for c in t):          # «,» spam is not a complaint
            continue
        same = [x for x, tx in ins[i:] if tx == t and _ts(x["ts"]) - _ts(r["ts"]) <= timedelta(minutes=5)]
        if len(same) >= 3:
            if not any(o[2] == "repeat" and o[3].startswith(f"«{t}»") for o in out):
                out.append(("WARN", r["ts"], "repeat", f"«{t}» sent {len(same)}x within 5 min"))

    # 3. Cancel pressed, work carried on
    for r in rows:
        d = r["data"] if isinstance(r["data"], dict) else {}
        if r["kind"] == "in" and str(d.get("button", "")).startswith("cancel:"):
            t0 = _ts(r["ts"])
            for o in outs:
                dt = _ts(o["ts"]) - t0
                if timedelta(0) <= dt <= timedelta(seconds=90):
                    if _DONE_CANCEL.search(otext(o)):
                        break
                    if dt > timedelta(seconds=1) and _WORK.search(otext(o)):
                        out.append(("ERROR", o["ts"], "cancel-ignored",
                                    f"after Cancel at {r['ts']}: {otext(o)[:70]!r}"))
                        break

    # 4. a stop word was taken as a note for the running task
    for o in outs:
        m = re.match(r"👌 (?:Noted|Учту): [«“](.*?)[»”]", otext(o))
        if m and m.group(1).strip(" .!").lower() in _STOP:
            out.append(("ERROR", o["ts"], "stop-as-steer", otext(o)[:60]))

    # 5. the same bot message twice in a row within 5 s (double send)
    prev = None
    for o in outs:
        t = otext(o)
        if (t and prev and t == otext(prev) and o["data"].get("method") == "sendMessage"
                and _ts(o["ts"]) - _ts(prev["ts"]) <= timedelta(seconds=5)):
            out.append(("WARN", o["ts"], "double-send", t[:60]))
        if o["data"].get("method") == "sendMessage":
            prev = o

    # 7. a Russian-speaking user got a message with no Russian in it
    #    (live 2026-09-27: every status said «⚙️ Starting…» to Russian users)
    cyr = sum(bool(re.search("[а-яё]", t, re.I)) for _r, t in ins)
    #    A Latin reply is RIGHT after a Latin-script question, an ask for
    #    English / a translation, or «answer in English from now on».
    if ins and cyr >= max(1, len(ins) // 2):
        import intent        # the model's read of each user line (agent/intent.py)
        last_in, pinned = "", False
        for r in rows:
            if r in [x for x, _ in ins]:
                last_in = (r["data"].get("text") or "").strip()
                _mode = intent.read(None, last_in)["language_mode"] if last_in else ""
                if _mode:
                    pinned = _mode == "en"
                continue
            if r not in outs:
                continue
            t = re.sub(r"<[^>]+>|https?://\S+|```.*?```", "", otext(r), flags=re.S)
            if not (len(re.findall(r"[A-Za-z]{3,}", t)) >= 1 and not re.search("[а-яё]", t, re.I)):
                continue
            _r = intent.read(None, last_in) if last_in else intent.FALLBACK
            if (pinned or not re.search("[а-яё]", last_in, re.I)
                    or _r["reply_language"] == "en" or _r["translate"]):
                continue
            out.append(("ERROR", r["ts"], "wrong-language", t.strip()[:60]))

    # 6. a button label answered by the model as if it were a question
    for i, (r, t) in enumerate(ins):
        if t in labels:
            nxt = [o for o in outs if timedelta(0) <= _ts(o["ts"]) - _ts(r["ts"]) <= timedelta(seconds=4)]
            if any(re.search(r"Starting|Начинаю|Writing a response|Пишу ответ", otext(o)) for o in nxt):
                out.append(("ERROR", r["ts"], "button-to-model", f"button «{t}» went to the model"))
    return out


def main(argv):
    since = None
    if "--since" in argv:
        since = argv[argv.index("--since") + 1]
        argv = [a for a in argv if a not in ("--since", since)]
    root = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "chat_logs")
    labels, findings = _labels(), []
    for f in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
        rows = []
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not since or r.get("ts", "") >= since:
                rows.append(r)
        chat = os.path.basename(f)[:-6]
        findings += [(lvl, chat, ts, rule, det) for lvl, ts, rule, det in audit_chat(rows, labels)]
    findings.sort(key=lambda x: (x[0] != "ERROR", x[2]))
    for lvl, chat, ts, rule, det in findings:
        print(f"{lvl:5} {ts} chat={chat} {rule:16} {det}")
    n_err = sum(f[0] == "ERROR" for f in findings)
    print(f"\n{n_err} error(s), {len(findings) - n_err} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
