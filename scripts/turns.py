"""Read the per-turn traces (agent/turn_trace.py) -- the debugging view.

  venv/Scripts/python scripts/turns.py            last 5 turns, warnings and key lines
  venv/Scripts/python scripts/turns.py -n 1 -v    the last turn with every event and LLM call
  venv/Scripts/python scripts/turns.py --grep rejected --chat <chat_id>
  venv/Scripts/python scripts/turns.py --stats    phase and LLM time across today's turns
  venv/Scripts/python scripts/turns.py --html      runtime/turns/dashboard.html: tokens and time per turn
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from turn_trace import _words  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Lines worth seeing without -v: guards, forcing, tools, rejections, rounds.
_KEY = ("Tool:", "forcing", "replacing", "rejected", "Round", "guard", "follow-up", "retry")


def load(day: str = ""):
    d = ROOT / "runtime" / "turns"
    files = [d / f"{day}.jsonl"] if day else sorted(d.glob("*.jsonl"))[-2:]
    out = []
    for f in files:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                try: out.append(json.loads(line))
                except ValueError: pass
    return out


def show(t, verbose):
    m, tot = t["meta"], t.get("totals", {})
    print("=" * 100)
    print(f"{t['ts']}  chat {m.get('chat')}  task {m.get('task')}  {m.get('outcome')}  "
          f"{m.get('seconds')}s | LLM {tot.get('llm_calls')} calls {tot.get('llm_seconds')}s "
          f"in {tot.get('tokens_in')} out {tot.get('tokens_out')} tok | tools {tot.get('tools')} "
          f"| warnings {tot.get('warnings')}")
    print("USER:", (m.get("text") or "").replace("\n", " ")[:300])
    if t.get("phases"):
        print("PHASES:", "  ".join(f"{p['name']} {p.get('s')}s" for p in t["phases"]))
    if verbose:
        for c in t.get("llm", []):
            print(f"  LLM +{c['t']:>6}s {c['s']:>5}s in {c.get('in')} out {c.get('out')} "
                  f"{c.get('finish')}  {c['purpose'][:60]!r}")
            for p in (c.get("messages") or [])[-1:]:
                for part in p["content"] if isinstance(p["content"], list) else [p["content"]]:
                    if isinstance(part, dict) and part.get("image"):
                        print(f"      image: {part['image']}")
            if (c.get("reply") or {}).get("content"):
                print("      -> " + c["reply"]["content"][:400].replace("\n", "\n         "))
    for m in t.get("media", []):
        print(f"  MEDIA {m['path']}")
    for e in t.get("events", []):
        if verbose or e["lvl"] in "WEC" or any(k in e["msg"] for k in _KEY):
            print(f"  {e['lvl']} +{e['t']:>6}s {e['src']:<14} {e['msg'][:220]}")
    a = t.get("answers", {})
    # Traces written before turn_trace dropped format-only drafts still carry them.
    if "draft" in a and _words(a["draft"]) != _words(a.get("final", "")):
        print("DRAFT (replaced):", a["draft"][:600 if not verbose else 6000])
    if "final" in a:
        print("FINAL:", a["final"][:400 if not verbose else 6000])


def stats(turns):
    from collections import defaultdict
    ph, n = defaultdict(float), defaultdict(int)
    for t in turns:
        for p in t.get("phases", []):
            ph[p["name"]] += p.get("s") or 0; n[p["name"]] += 1
    print(f"{len(turns)} turns, {sum(t['meta'].get('seconds') or 0 for t in turns):.0f}s total, "
          f"LLM {sum(t.get('totals', {}).get('llm_seconds') or 0 for t in turns):.0f}s, "
          f"tokens out {sum(t.get('totals', {}).get('tokens_out') or 0 for t in turns)}")
    for k in sorted(ph, key=ph.get, reverse=True)[:15]:
        print(f"  {k:<40} {ph[k]:>8.1f}s  x{n[k]}  avg {ph[k] / n[k]:.1f}s")


def html(turns):
    import html as H
    top = max([t.get("totals", {}).get("tokens_in") or 0 for t in turns] + [1])
    rows = []
    for t in reversed(turns):
        m, tot = t["meta"], t.get("totals", {})
        w = 100 * (tot.get("tokens_in") or 0) / top
        ph = " · ".join(f"{p['name']} {p.get('s')}s" for p in t.get("phases", []))
        warn = "<br>".join(H.escape(e["msg"][:200]) for e in t.get("events", []) if e["lvl"] in "WEC")
        rows.append(f"<tr><td>{t['ts']}</td><td>{m.get('chat')}</td><td>{m.get('outcome')}</td>"
                    f"<td>{m.get('seconds')}s</td><td><div class=b style='width:{w:.0f}%'></div>"
                    f"{tot.get('tokens_in')}/{tot.get('tokens_out')} · {tot.get('llm_calls')} calls</td>"
                    f"<td>{H.escape((m.get('text') or '')[:120])}<div class=p>{H.escape(ph)}</div>"
                    f"<div class=w>{warn}</div></td></tr>")
    out = ROOT / "runtime" / "turns" / "dashboard.html"
    out.write_text("<meta charset=utf-8><style>body{font:13px sans-serif;background:#111;color:#ddd}"
                   "td{border-bottom:1px solid #333;padding:4px;vertical-align:top}.b{height:6px;background:#4a8}"
                   ".p{color:#888}.w{color:#e96}</style><table><tr><th>time<th>chat<th>outcome<th>sec"
                   "<th>tokens in/out<th>ask</tr>" + "".join(rows) + "</table>", encoding="utf-8")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--chat", default="")
    ap.add_argument("--grep", default="")
    ap.add_argument("--day", default="")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--html", action="store_true")
    a = ap.parse_args()
    turns = load(a.day)
    if a.chat:
        turns = [t for t in turns if str(t["meta"].get("chat")) == a.chat]
    if a.grep:
        turns = [t for t in turns if a.grep.lower() in json.dumps(t, ensure_ascii=False).lower()]
    if a.html:
        return html(turns)
    if a.stats:
        return stats(turns)
    for t in turns[-a.n:]:
        show(t, a.v)


if __name__ == "__main__":
    main()
