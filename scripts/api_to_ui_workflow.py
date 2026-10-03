"""Convert an API-format ComfyUI graph into an editable UI workflow.

video.py posts API-format JSON (node id -> {class_type, inputs}) because that is
what POST /prompt consumes. The graph editor wants the other shape: a `nodes`
list with positions, explicit `links`, and `widgets_values` ordered exactly as
the node declares its inputs. Dropping an API file on the canvas does not give
you a workflow you can meaningfully rearrange, which is the whole point of
opening it by hand.

Widget order comes from the live server's /object_info, so this cannot drift
from whatever nodes are actually installed.

Run:
  venv/Scripts/python.exe scripts/api_to_ui_workflow.py workflow_video_h3.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import requests

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8000")

# Widgets the UI silently appends after a given input. The seed control is the
# classic one: the editor stores seed AND its control_after_generate mode, so a
# file without it loads with every later widget shifted by one slot.
_CONTROL_AFTER = {"seed", "noise_seed"}


def object_info(cls: str) -> dict:
    r = requests.get(f"{COMFY_URL}/object_info/{cls}", timeout=15)
    r.raise_for_status()
    return (r.json() or {}).get(cls) or {}


def convert(api: dict) -> dict:
    infos = {}
    for nid, node in api.items():
        cls = node["class_type"]
        if cls not in infos:
            infos[cls] = object_info(cls)

    links: list = []
    link_id = 1
    nodes_out = []
    # Grid layout: wide columns, so a 12-node graph opens readable instead of
    # stacked at the origin.
    col_w, row_h = 420, 260
    order = list(api.keys())
    pos = {nid: (300 + (i % 4) * col_w, 100 + (i // 4) * row_h)
           for i, nid in enumerate(order)}

    # First pass: inputs/outputs and the links between them.
    inputs_by_node: dict[str, list] = {n: [] for n in order}
    outputs_by_node: dict[str, list] = {n: [] for n in order}

    for nid in order:
        node = api[nid]
        info = infos[node["class_type"]]
        req = (info.get("input") or {}).get("required") or {}
        opt = (info.get("input") or {}).get("optional") or {}
        spec = {**req, **opt}

        for name, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and str(val[0]) in api:
                src_id, src_slot = str(val[0]), int(val[1])
                entry = spec.get(name)
                typ = entry[0] if isinstance(entry, (list, tuple)) and entry else "*"
                if isinstance(typ, list):
                    typ = "COMBO"
                links.append([link_id, int(src_id), src_slot, int(nid),
                              len(inputs_by_node[nid]), typ])
                inputs_by_node[nid].append(
                    {"name": name, "type": typ, "link": link_id})
                outputs_by_node.setdefault(src_id, [])
                link_id += 1

    for nid in order:
        info = infos[api[nid]["class_type"]]
        out_names = info.get("output_name") or info.get("output") or []
        out_types = info.get("output") or []
        outs = []
        for slot, t in enumerate(out_types):
            name = out_names[slot] if slot < len(out_names) else str(t)
            if isinstance(name, list):
                name = str(t)
            mine = [l[0] for l in links if l[1] == int(nid) and l[2] == slot]
            outs.append({"name": name, "type": t, "links": mine or None,
                         "slot_index": slot})
        outputs_by_node[nid] = outs

    for nid in order:
        node = api[nid]
        info = infos[node["class_type"]]
        req = (info.get("input") or {}).get("required") or {}
        # Widgets are the required inputs that are NOT wired, in declared order.
        widgets = []
        linked = {i["name"] for i in inputs_by_node[nid]}
        for name in req:
            if name in linked:
                continue
            if name not in node.get("inputs", {}):
                continue
            widgets.append(node["inputs"][name])
            if name in _CONTROL_AFTER:
                widgets.append("fixed")
        x, y = pos[nid]
        nodes_out.append({
            "id": int(nid),
            "type": node["class_type"],
            "pos": [x, y],
            "size": [380, 120],
            "flags": {},
            "order": order.index(nid),
            "mode": 0,
            "inputs": inputs_by_node[nid],
            "outputs": outputs_by_node[nid],
            "properties": {"Node name for S&R": node["class_type"]},
            "widgets_values": widgets,
        })

    return {
        "id": "h3-video",
        "revision": 0,
        "last_node_id": max(int(n) for n in order),
        "last_link_id": link_id - 1,
        "nodes": nodes_out,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("api_json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = os.path.abspath(args.api_json)
    with open(src, encoding="utf-8") as fh:
        api = json.load(fh)
    if "nodes" in api and isinstance(api.get("nodes"), list):
        print(f"{src} is already a UI workflow — nothing to do.")
        return 0

    ui = convert(api)
    out = args.out or src.replace(".json", "_ui.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ui, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out}  ({len(ui['nodes'])} nodes, {len(ui['links'])} links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
