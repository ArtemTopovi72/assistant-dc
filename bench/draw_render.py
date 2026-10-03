"""Actually draw the scenarios. Two phases, because they do not fit together.

draw_scenarios.py measures the LAYOUT and never renders -- deliberately, that is
the machine-checkable half. This is the other half: take the layout each
scenario ends with and put it through Ideogram, so the arrangement can be looked
at as a picture.

Why two phases: LM Studio holds ~20 GB of a 24 GB card and Ideogram 4 wants
roughly as much again (two UNETs plus a text encoder). Rendering while the LLM
is resident thrashes, so every layout is planned FIRST, with the model loaded,
then the model is unloaded and every render happens with the card to itself.
`--keep-llm` skips the unload for a machine that can hold both.

The model is reloaded at the end, because leaving the assistant without its LLM
is a worse outcome than a slow bench.

Usage:
    venv/Scripts/python.exe bench/draw_render.py [--case move_cat] [--out DIR]
                                                 [--keep-llm] [--reload]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_scenarios as DS                    # noqa: E402
import draw_agent as DA                       # noqa: E402
from tc_run import setup                       # noqa: E402
import ideogram as IG                          # noqa: E402
import config as CFG                           # noqa: E402


# The critic condemns almost every picture, including ones that are plainly
# fine -- it objects to a dim kitchen at 79/255 mean brightness as "extremely
# dark". Checked directly: shown the same file it describes the contents
# accurately, so it is seeing, not hallucinating; it is simply a harsh grader.
# A verdict of "BAD" on 14 of 14 carries no information.
#
# What DOES carry information is WHICH complaint. Its collage complaints match
# what is visible by eye every time, so the problems are bucketed and counted,
# and the bench reports the buckets rather than a pass rate nobody can act on.
_BUCKETS = (
    ("collage", r"collage|disconnected|incoherent|split into|separate (?:panels|"
                r"images)|grid of|multiple (?:panels|photos)|panel"),
    ("scale",   r"much larger|too large|much bigger|oversized|covers most|"
                r"occupies most|larger than requested"),
    ("missing", r"missing|not present|absent|does not appear|no (?:cat|box|mug|"
                r"lamp|table|sofa)"),
    ("place",   r"not placed|wrong position|positioned differently|shifted|"
                r"not where"),
    ("light",   r"dark|underexposed|low contrast|hard to see|barely visible"),
    ("text",    r"text|lettering|words|logo|caption|sign"),
)


def _bucket(problem: str) -> str:
    import re as _re
    for name, pattern in _BUCKETS:
        if _re.search(pattern, problem or "", _re.IGNORECASE):
            return name
    return "other"


def _free_comfy() -> str:
    """Ask ComfyUI to drop its models before the LLM comes back.

    Without this the judge phase stalls: ComfyUI keeps ~17 GB resident after a
    render, `lms load` cannot fit 20.5 GB beside it, and the load times out
    while reporting a failure that is really a memory shortage. One card, two
    tenants -- whoever goes second has to be let in.
    """
    import requests
    url = getattr(CFG, "COMFY_URL", "http://127.0.0.1:8000")
    try:
        # 180s, not 30: /free is answered on the same thread that finishes the
        # current job, so a request sent while the last picture is still being
        # written just times out -- which is what happened, and the run then
        # spent two ten-minute `lms load` attempts failing to fit 20 GB beside
        # a renderer that had never let go.
        r = requests.post(url + "/free",
                          json={"unload_models": True, "free_memory": True},
                          timeout=180)
        note = "freed the card (HTTP %s)" % r.status_code
    except Exception as exc:
        note = "could not free ComfyUI: %s" % exc
    return note + "; " + _wait_for_vram()


def _wait_for_vram(need_mib: int = 21000, secs: int = 120) -> str:
    """Block until the card actually has room, or say that it never did.

    Asking ComfyUI to free is not the same as the memory being back: the driver
    releases it a moment later. Loading into a card that is still full fails
    slowly and reports the wrong cause, so wait for the number instead of
    trusting the HTTP 200.
    """
    deadline = time.time() + secs
    free = -1
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
                stdin=subprocess.DEVNULL)
            free = int((out.stdout or "0").strip().splitlines()[0])
        except Exception:
            return "could not read VRAM"
        if free >= need_mib:
            return "%d MiB free" % free
        time.sleep(5)
    return "still only %d MiB free after %ss" % (free, secs)


def _lms(*args) -> str:
    try:
        # 600s, not 180: loading a 20 GB model off this disk takes longer than
        # three minutes, and the judge phase died reporting a load failure that
        # had not actually failed -- it was still going.
        r = subprocess.run(["lms", *args], capture_output=True, text=True,
                           timeout=600, stdin=subprocess.DEVNULL)
        return (r.stdout or r.stderr or "").strip()
    except Exception as exc:
        return "lms %s failed: %s" % (" ".join(args), exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--keep-llm", action="store_true")
    ap.add_argument("--reload", default="", help="model id to load back afterwards")
    ap.add_argument("--any-model", action="store_true",
                    help="plan with whatever LM Studio happens to be serving")
    ap.add_argument("--no-render", action="store_true",
                    help="judge pictures that are already there, "
                         "without drawing them again")
    ap.add_argument("--judge", action="store_true",
                    help="after rendering, reload the LLM and ask the "
                         "critic whether each picture matches its layout")
    ap.add_argument("--from", dest="from_dir", default="",
                    help="re-render the layouts a previous run saved there")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()

    out = Path(args.out or (Path("runtime") / ("draw_render_%d" % time.time())))
    out.mkdir(parents=True, exist_ok=True)

    cases = [c for c in DS.CASES if not args.case or c["id"] in args.case]
    if args.from_dir:
        # Re-render layouts a previous run saved. This is how a change to the
        # geometry repair or the caption gets checked in PIXELS without asking
        # the model for a fresh layout: the same boxes go in and only the
        # repair differs, so anything that moves is the change, not sampling.
        plans = []
        for f in sorted(Path(args.from_dir).glob("*.layout.json")):
            cid = f.name[:-len(".layout.json")]
            if args.case and cid not in args.case:
                continue
            fixed, notes = DA.auto_fix_geometry(
                json.loads(f.read_text(encoding="utf-8")))
            plans.append(dict(id=cid, ok=True, layout=fixed,
                              why="re-rendered from " + str(f)))
            print("[load] %-22s %s" % (cid, "; ".join(notes)[:70] or "no repair"))
        return _render(plans, out, args)


    _graph, ctx, _img = setup()
    model = args.reload or CFG.MODEL_NAME

    # WHICH model answered is part of the result. Phase 2 unloads everything,
    # and LM Studio then just-in-time loads whatever it likes on the next
    # request -- a later run of this bench planned its layouts with
    # mantella-gemma4-a4b and reported two planning failures that belong to
    # that model, not to the house one. Say so loudly rather than publishing a
    # number about a model nobody chose.
    served = str(getattr(ctx, "model_name", "") or "")
    if served != CFG.MODEL_NAME:
        print("!! serving %r, expected %r -- planning results would be about "
              "the wrong model." % (served, CFG.MODEL_NAME))
        if not args.any_model:
            print("   load it (`lms load %s`) or pass --any-model." % CFG.MODEL_NAME)
            return 2

    # ── phase 1: every layout, with the LLM loaded ──────────────────────────
    plans = []
    for case in cases:
        row = DS.run_case(ctx, case)
        plans.append(row)
        print("[plan] %-22s %s  %s" % (
            row["id"], "ok " if row["ok"] else "BAD", row["why"][:70]))
        (out / (row["id"] + ".layout.json")).write_text(
            json.dumps(row["layout"], ensure_ascii=False, indent=2),
            encoding="utf-8")
        (out / (row["id"] + ".trace.json")).write_text(
            json.dumps(row["trace"], ensure_ascii=False, indent=2),
            encoding="utf-8")

    return _render(plans, out, args, unload=True)


def _render(plans, out, args, unload=False):
    # ── phase 2: the card to itself ─────────────────────────────────────────
    if unload and not args.keep_llm:
        print("\nunloading the LLM so Ideogram has the card: " + _lms("unload", "--all"))

    rows = []
    if getattr(args, "no_render", False):
        # Judge what is already on disk. Re-rendering to look at a picture
        # costs three minutes of GPU and gives a DIFFERENT picture, which is
        # not the one anybody wanted judged.
        for row in plans:
            hit = [p for p in Path(out).glob(row["id"] + ".*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
            rows.append((row["id"], bool(hit), 0.0,
                         str(hit[0]) if hit else "no picture on disk"))
            print("[have] %-22s %s" % (row["id"], rows[-1][3]))
    else:
        for row in plans:
            caption = IG.layout_to_caption(row["layout"])
            (out / (row["id"] + ".caption.json")).write_text(
                json.dumps(caption, ensure_ascii=False, indent=2), encoding="utf-8")
            t0 = time.perf_counter()
            try:
                # ctx=None on purpose: nothing in the render path may call the LLM
                # here -- it has just been unloaded, and a planner call would either
                # hang or quietly JIT-load 20 GB back onto the card mid-render.
                # Through the SAME collage defence the product uses. Rendering
                # with IG.generate directly measured something the user never
                # gets: 15 of 53 renders from earlier runs of this bench came
                # back as collages or cut-outs, and the ladder is what removes
                # them. A bench that skips it reports a worse pipeline than the
                # one that ships.
                path = DA.render_without_collage(
                    None, row["layout"], width=args.width, height=args.height,
                    seed=row.get("seed") or 12345)
                why = path or "the renderer returned nothing"
                still_collage = bool(path) and DA.looks_like_collage(path)
            except Exception as exc:
                path, why, still_collage = None, "%s: %s" % (type(exc).__name__, exc), False
            dt = time.perf_counter() - t0
            if path:
                dest = out / (row["id"] + Path(path).suffix)
                try:
                    dest.write_bytes(Path(path).read_bytes())
                    # Appended AFTER the copy: the first version of this line set
                    # the marker and then had it overwritten by the destination
                    # path, so a residual collage was reported as a clean render.
                    why = str(dest) + ("   [STILL A COLLAGE]" if still_collage else "")
                except Exception as exc:
                    why = "%s (could not copy: %s)" % (path, exc)
            rows.append((row["id"], bool(path), dt, why))
            print("[draw] %-22s %6.1fs  %s" % (row["id"], dt, why[:90]))

        if unload and not args.keep_llm and args.reload:
            # Order matters: the renderer is still holding the card here.
            print(_free_comfy())
            # --yes, or the load waits for a confirmation that cannot arrive:
            # stdin is DEVNULL, so it sat there until the 600s timeout and the
            # bench walked away leaving the machine WITHOUT its model. That cost
            # a song render, a deck build and a bench start today before the
            # cause was found.
            print("reloading %s: %s"
                  % (args.reload, _lms("load", args.reload, "--yes")))

    # -- phase 3: does the PICTURE match the layout? -------------------------
    # The layout score is machine-checkable and the pixels are not, which is why
    # draw_scenarios stops at the layout. But "the boxes were right and the
    # picture still isn't" is exactly the class of defect that only shows up
    # here -- a mug that floats, an invented signboard -- so this asks the same
    # critic the drawing loop uses. Its verdict is EVIDENCE, not a gate: a critic
    # that cannot see reports ok=True, and a bench must not read that as a pass.
    verdicts = []
    if args.judge:
        print()
        if args.reload:
            print(_free_comfy())
            print("reloading %s to judge: %s"
                  % (args.reload, _lms("load", args.reload, "--yes")))
        for row, (cid, ok, _dt, why) in zip(plans, rows):
            if not ok:
                continue
            from tc_run import _make_ctx, _served_model
            try:
                jctx = _make_ctx(_served_model())
                v = DA.critique(jctx, why, row["layout"])
            except SystemExit as exc:
                # _served_model exits when nothing is loaded. A judge that
                # cannot see is not a verdict about the picture, and the run
                # must not end as if it were.
                print("  no model to judge with (%s) — pictures kept, unjudged"
                      % exc)
                break
            except Exception as exc:
                v = {"ok": None, "source": "failed", "problems": [str(exc)]}
            verdicts.append((cid, v))
            mark = {True: "ok  ", False: "BAD ", None: "??  "}.get(v.get("ok"), "??  ")
            print("[look] %-22s %s%s  %s" % (
                cid, mark, v.get("source", ""),
                "; ".join(str(p) for p in (v.get("problems") or []))[:80]))


    if verdicts:
        from collections import Counter
        seen = Counter()
        for cid, v in verdicts:
            for kind in {_bucket(p) for p in (v.get("problems") or [])}:
                seen[kind] += 1
        print()
        print("  what the critic complained about, by picture:")
        for kind, n in seen.most_common():
            print("    %-8s %d/%d" % (kind, n, len(verdicts)))

    drawn = sum(1 for _i, ok, _d, _w in rows if ok)
    print("\n" + "=" * 72)
    print("  layouts %d/%d ok, rendered %d/%d" % (
        sum(1 for r in plans if r["ok"]), len(plans), drawn, len(rows)))
    print("  " + str(out))
    return 0 if drawn == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
