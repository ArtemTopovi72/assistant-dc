"""Offline retrieval recall: is the expected tool among what select_tools sends?

    venv/Scripts/python bench/tool_recall.py

No model call except embeddings. For each case of tc_cases + tc_para, runs
select_tools with TOOLS_EMBED off and on and reports recall of `expect` and
the mean schema count (payload). Gate cases (expect=None) count only for size.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tools                                   # noqa: E402
import tool_retrieval as R                     # noqa: E402
from bench.tc_cases import CASES               # noqa: E402
from bench.tc_para import PARA                 # noqa: E402


def run(mode):
    os.environ["TOOLS_EMBED"] = mode
    rows = {}
    for c in CASES + PARA:
        sel = [R._name(s) for s in R.select_tools(c["text"], tools.TOOL_SCHEMAS)]
        cue = any(rx.search(c["text"]) for rx in R._CUE_RE.values())
        rows[c["id"]] = (c.get("expect"), sel, cue)
    return rows


def main():
    res = {m: run(m) for m in ("0", "1")}
    for group, ids in (("tc_cases", [c["id"] for c in CASES]), ("para", [c["id"] for c in PARA])):
        print(f"\n== {group}")
        for m in ("0", "1"):
            r = res[m]
            need = [i for i in ids if r[i][0]]
            hit = [i for i in need if r[i][0] in r[i][1]]
            size = sum(len(r[i][1]) for i in ids) / len(ids)
            print(f"  TOOLS_EMBED={m}: recall {len(hit)}/{len(need)}, mean schemas {size:.1f}")
    print("\npara detail (off -> on):")
    for c in PARA:
        a, b = res["0"][c["id"]], res["1"][c["id"]]
        flag = "CUE!" if a[2] else "    "
        print(f"  {flag} {c['id']:<18} {c['expect']:<20} {'Y' if a[0] in a[1] else '-'} -> "
              f"{'Y' if b[0] in b[1] else '-'}  {sorted(set(b[1]) - set(a[1]))}")
    miss = [c["id"] for c in CASES if c.get("expect") and c["expect"] in res["0"][c["id"]][1]
            and c["expect"] not in res["1"][c["id"]][1]]
    print("\ntc_cases lost by embedding:", miss or "none")


if __name__ == "__main__":
    main()
