r"""SkillOpt-style optimizer for the tool descriptions, with RRSI regularisation.

    venv/Scripts/python bench/skillopt.py --iters 8 [--reps 2]

The "skill" is the tool-description prose the model routes on. The model under
test stays frozen; a separate optimizer call (the same served model, as a
different role) turns scored rollouts into BOUNDED edits:

  * learning rate  -- at most `budget` edits per step, annealed 2 -> 1
  * rejected buffer -- every rejected edit is shown back so it is not re-proposed
  * strict accept  -- an edit is kept only if the HELD-OUT score strictly rises
                      and the train score does not fall

Train = bench/tc_cases (route/args/chain/recover/safety + gate).
Held-out = bench/tc_para (paraphrases with none of the cue words) + the gate
cases again, so an edit that makes a tool greedy is caught.

RRSI critic: an edit that quotes 3+ consecutive words from ANY case text is
benchmark memorisation and is rejected before it costs an evaluation; so is an
edit that grows a description by more than 300 chars (pruner).

Nothing is written to tool_descriptions.py. The winning edits go to
runtime/skillopt/best_edits.json; applying them is a separate, reviewed step.
"""
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.tc_run import setup, run_case          # noqa: E402
from bench.tc_cases import CASES                  # noqa: E402
from bench.tc_para import PARA                    # noqa: E402
import requests                                   # noqa: E402
import config                                     # noqa: E402
import utils                                      # noqa: E402

OUT = Path("runtime/skillopt")
TRAIN = [c for c in CASES if not c.get("image")]
GATE = [c for c in CASES if c["cat"] == "gate"]
VAL = PARA + GATE
ALL_TEXT = [c["text"].lower() for c in CASES + PARA]


def _desc(schemas):
    return {s["function"]["name"]: s["function"].get("description", "") for s in schemas}


def apply_edits(base, edits):
    s = copy.deepcopy(base)
    by = {x["function"]["name"]: x["function"] for x in s}
    for e in edits:
        f = by.get(e["tool"])
        if f is None:
            raise ValueError(f"no tool {e['tool']}")
        d = f.get("description", "")
        if e["op"] == "append":
            f["description"] = d.rstrip() + " " + e["new"].strip()
        elif e["op"] == "replace":
            if e["old"] not in d:
                raise ValueError("old text not in description")
            f["description"] = d.replace(e["old"], e["new"], 1)
        else:
            raise ValueError(f"bad op {e['op']}")
    return s


def critic(edit, base_desc):
    """RRSI: reject benchmark-specific or bloated edits before evaluating them."""
    new = (edit.get("new") or "").lower()
    words = re.findall(r"\w+", new)
    for i in range(len(words) - 2):
        tri = " ".join(words[i:i + 3])
        if any(tri in " ".join(re.findall(r"\w+", t)) for t in ALL_TEXT):
            return f"quotes case text '{tri}'"
    grow = len(edit.get("new") or "") - len(edit.get("old") or "")
    if grow > 300:
        return f"grows description by {grow} chars"
    if edit.get("tool") not in base_desc:
        return "unknown tool"
    return None


def evaluate(graph, ctx, stub, cases, schemas, reps):
    rows = []
    for _ in range(reps):
        for c in cases:
            r = run_case(graph, ctx, c, stub, schemas=schemas)
            rows.append({k: r[k] for k in ("id", "verdict", "why", "calls", "expect_tool")})
    return sum(r["verdict"] == "PASS" for r in rows), rows


def propose(model, fails, descs, rejected, budget):
    tools_seen = set()
    for f in fails:
        tools_seen.update([f.get("expect_tool") or ""] + (f.get("calls") or []))
    tools_seen.discard("")
    shown = {t: descs[t] for t in sorted(tools_seen) if t in descs}
    prompt = (
        "You improve the DESCRIPTIONS of tools an assistant routes on. The model reading them is "
        "frozen; only the text changes. Below are routing failures (user text is Russian), the "
        "current descriptions of the tools involved, and edits already rejected.\n\n"
        f"FAILURES:\n{json.dumps(fails[:14], ensure_ascii=False, indent=0)}\n\n"
        f"DESCRIPTIONS:\n{json.dumps(shown, ensure_ascii=False, indent=0)}\n\n"
        f"REJECTED (do not repeat):\n{json.dumps(rejected[-12:], ensure_ascii=False)}\n\n"
        f"Propose at most {budget} edit(s) that fix a GENERAL confusion (a class of requests), "
        "not one example. Never quote the user texts. Prefer sharpening when-to-use / "
        "when-NOT-to-use boundaries between the confused tools. Keep edits short (under 200 chars).\n"
        'Answer ONLY JSON: {"edits":[{"tool":"name","op":"append"|"replace","old":"exact '
        'substring for replace, else empty","new":"text","why":"short"}]}')
    r = requests.post(f"{config.LM_STUDIO_BASE}/v1/chat/completions", timeout=300, json={
        "model": model, "temperature": 0.7, "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": "<|channel>thought\n<channel|>"}]}).json()
    txt = r["choices"][0]["message"].get("content") or ""
    data = utils.safe_json_from_llm(txt) or {}
    return (data.get("edits") or [])[:budget] if isinstance(data, dict) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "log.jsonl", "a", encoding="utf-8")
    graph, ctx, stub = setup()
    base = copy.deepcopy(graph.TOOL_SCHEMAS)
    kept, rejected = [], []

    cur = base
    tr, tr_rows = evaluate(graph, ctx, stub, TRAIN, cur, 1)
    va, _ = evaluate(graph, ctx, stub, VAL, cur, a.reps)
    print(f"baseline train {tr}/{len(TRAIN)}  val {va}/{len(VAL) * a.reps}", flush=True)
    json.dump({"train": tr_rows}, open(OUT / f"baseline_{ctx.model_name.replace(':', '_')}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.write(json.dumps({"ts": time.time(), "baseline": [tr, va]}) + "\n"); log.flush()

    for it in range(a.iters):
        budget = 2 if it < a.iters // 2 else 1                     # annealed LR
        fails = [r for r in tr_rows if r["verdict"] != "PASS"]
        if not fails:
            print("no train failures left"); break
        edits = propose(ctx.model_name, fails, _desc(cur), rejected, budget)
        good = []
        for e in edits:
            why = critic(e, _desc(cur))
            if why:
                rejected.append({**e, "rejected": f"critic: {why}"})
            else:
                good.append(e)
        if not good:
            print(f"[{it}] nothing survived the critic ({len(edits)} proposed)", flush=True)
            continue
        try:
            cand = apply_edits(cur, good)
        except ValueError as exc:
            rejected += [{**e, "rejected": str(exc)} for e in good]
            print(f"[{it}] edit did not apply: {exc}", flush=True)
            continue
        ctr, ctr_rows = evaluate(graph, ctx, stub, TRAIN, cand, 1)
        cva, _ = evaluate(graph, ctx, stub, VAL, cand, a.reps)
        ok = cva > va and ctr >= tr
        print(f"[{it}] train {tr}->{ctr}  val {va}->{cva}  {'ACCEPT' if ok else 'reject'}  "
              f"{[(e['tool'], e['new'][:70]) for e in good]}", flush=True)
        log.write(json.dumps({"ts": time.time(), "it": it, "edits": good, "train": [tr, ctr],
                              "val": [va, cva], "accept": ok}, ensure_ascii=False) + "\n"); log.flush()
        if ok:
            cur, tr, va, tr_rows = cand, ctr, cva, ctr_rows
            kept += good
            (OUT / "best_edits.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
        else:
            rejected += [{**e, "rejected": f"val {va}->{cva}, train {tr}->{ctr}"} for e in good]
    print(f"\nfinal train {tr}/{len(TRAIN)}  val {va}/{len(VAL) * a.reps}  kept {len(kept)} edit(s)")


if __name__ == "__main__":
    main()
