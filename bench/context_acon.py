"""ACON-style refinement of the compaction rules from failed journeys (arXiv 2510.00615).

ACON does not fine-tune anything: it runs the agent, looks at the tasks that
failed AFTER a compaction, asks a model what the compressed context was missing,
and folds that into the natural-language guideline the compressor follows. Here
the guideline is runtime/context_guidelines.txt, appended to
context_v2.DELTA_PROMPT on every compaction.

  1. run journeys:      python bench/live_journeys.py --mega
  2. refine:            python bench/context_acon.py runtime/live_drive/<stamp>
  3. A/B:               CONTEXT_GUIDELINES=nonexistent  vs default, same journeys,
                        then bench/context_cost.py --compare A B

Only steps that failed AFTER a compaction happened in that chat are used: a
failure with the full transcript in view says nothing about the compressor.
The model proposes at most 5 rules; each must name WHAT to keep, and a rule
that asks to keep everything is refused (it is the no-compression rule).
Rules are merged into the file, de-duplicated, capped at 20 lines; the
proposal, the evidence and the old file are saved beside it for review.
"""
import glob
import json
import os
import re
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GUIDE = os.path.join(ROOT, "runtime", "context_guidelines.txt")
LMS = os.getenv("LM_STUDIO_BASE", "http://127.0.0.1:1234/v1")
MAX_RULES = 20

PROMPT = """You tune the memory compressor of a chat assistant.
The compressor folds old turns into a structured working memory (goal,
decisions, files & pictures, open requests, user preferences) and archives the
raw turns under ids the assistant can recall.
Below are conversation steps that FAILED after such a fold, with the memory
the assistant had at that moment. For each failure decide whether the memory
was missing something the next step needed. Then write at most 5 short rules
for the compressor, each naming a concrete kind of information to keep
(e.g. "keep the exact wording of any text the user asked to put on a picture").
Never write a rule that says to keep everything. Current rules:
{rules}
Return ONLY JSON: {{"analysis": ["one line per failure"], "rules": ["..."]}}"""


def _load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def failures(run_dir):
    import context_v2 as C
    report = _load(os.path.join(run_dir, "journeys.json")) or []
    sessions = {}
    for f in glob.glob(os.path.join(run_dir, "**", "tg_sessions*.json"), recursive=True):
        obj = _load(f) or {}
        for k, v in (obj.items() if isinstance(obj, dict) else []):
            if isinstance(v, dict):
                sessions[str(k)] = v.get("history") or v.get("messages") or []
    out = []
    for r in report:
        chat = str(r.get("chat_id") or (900000 + int(r.get("num") or 0)))
        hist = sessions.get(chat) or []
        mem = next((m.get("content") for m in hist if m.get("role") == "system"
                    and str(m.get("content") or "").startswith(C.MEMORY_MARKER.rstrip())), "")
        if not mem:
            continue                      # never compacted: not the compressor's failure
        for s in r.get("steps", []):
            if not s.get("ok"):
                out.append({"journey": r.get("name"), "step": s.get("label"),
                            "problems": s.get("problems"), "bot": (s.get("text") or "")[:600],
                            "memory": mem[:2500]})
    return out


def propose(fails, rules):
    body = json.dumps(fails[:12], ensure_ascii=False, indent=1)
    model = os.getenv("ACON_MODEL") or requests.get(f"{LMS}/models", timeout=10).json()["data"][0]["id"]
    r = requests.post(f"{LMS}/chat/completions", timeout=600, json={
        "model": model, "temperature": 0.2, "max_tokens": 1200,
        "messages": [{"role": "system", "content": PROMPT.format(rules="\n".join(rules) or "(none)")},
                     {"role": "user", "content": body}]})
    from utils import safe_json_from_llm
    return safe_json_from_llm(r.json()["choices"][0]["message"]["content"] or "") or {}


def merge(rules, new):
    seen = {re.sub(r"\W+", " ", x.lower()).strip() for x in rules}
    out = list(rules)
    for x in new or []:
        x = " ".join(str(x).split())[:200]
        k = re.sub(r"\W+", " ", x.lower()).strip()
        if not x or k in seen or re.search(r"\b(everything|all information|всё|все подряд)\b", x, re.I):
            continue
        seen.add(k); out.append(x)
    return out[-MAX_RULES:]


def main(argv):
    if not argv:
        print(__doc__); return
    run_dir = argv[0]
    fails = failures(run_dir)
    rules = [l.strip("- ").strip() for l in open(GUIDE, encoding="utf-8").read().splitlines()
             if l.strip()] if os.path.exists(GUIDE) else []
    print(f"{len(fails)} failed step(s) after a compaction in {run_dir}")
    if not fails:
        return
    prop = propose(fails, rules)
    merged = merge(rules, prop.get("rules"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(GUIDE), exist_ok=True)
    json.dump({"run": run_dir, "failures": fails, "proposal": prop, "old": rules, "new": merged},
              open(os.path.join(os.path.dirname(GUIDE), f"context_acon_{stamp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(GUIDE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(f"- {x}" for x in merged) + "\n")
    for a in prop.get("analysis") or []:
        print("  ·", a)
    print(f"rules: {len(rules)} -> {len(merged)}  ({GUIDE})")


if __name__ == "__main__":
    main(sys.argv[1:])
