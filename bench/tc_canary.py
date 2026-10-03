r"""Canary-tool bench: put plausible-but-wrong DECOYS next to the real tools
and measure how often the model reaches for the decoy instead.

    .\venv\Scripts\python.exe bench\tc_canary.py [--reps 2] [--cat route]

The decoys are deliberately written the way a second developer would write
them if they had never read tool_descriptions.py: short, generic, no "use this
INSTEAD of that" steering. That makes each one a probe of exactly one real
tool's prose. If `web_lookup` — eleven words with no routing guidance — keeps
stealing calls from `search`, then `search`'s description is not actually what
is winning those calls, and the routing is far more fragile than the main
bench's pass rate suggests.

A decoy answers with a plausible SUCCESS string, so a stolen call looks to the
model exactly like a real one and the turn continues normally — we measure the
first choice, not the recovery.

Two numbers per decoy:
  steal  -- of the cases where the decoy's REAL partner was the right answer,
            how often the decoy was called instead/as well. This is the number
            that indicts a description.
  stray  -- of every OTHER case, how often the decoy got called anyway. A high
            stray rate means the decoy is simply attractive noise, and says
            more about the model than about the paired description.

Live LM Studio only; every tool (real and decoy) is a stub.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.tc_run import setup, run_case          # noqa: E402
from bench.tc_mocks import ToolRecorder           # noqa: E402
from bench.tc_cases import cases_for              # noqa: E402


class Decoy:
    """One plausible-but-wrong tool, shadowing exactly one real tool."""

    def __init__(self, name, shadows, desc, params, image=False, result=None):
        self.name = name
        self.shadows = shadows
        self.desc = desc
        self.params = params
        self.image = image          # only offer it when a picture is loaded
        self.result = result or "Done."

    @property
    def schema(self):
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.desc,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in self.params},
                "required": list(self.params),
            },
        }}

    def stub(self, state, args):
        # A decoy that "works": it sets the same state keys a successful real
        # call would, so the loop's post-call guards do not flag the turn and
        # the model never learns it picked the wrong door.
        if self.image:
            state["image_path"] = f"outputs/_bench_decoy_{self.name}.png"
            state["image_status"] = "ok"
        return self.result


DECOYS = [
    Decoy("web_lookup", "search",
          "Look up information on the internet and return relevant results.",
          ["query"], result="Results: A, B, C."),
    Decoy("research_report", "deep_research",
          "Research a topic online and write it up.",
          ["topic"], result="Report written."),
    Decoy("make_picture", "generate_image",
          "Create a picture from a text description.",
          ["description"], image=True,
          result="Picture created and shown to the user."),
    Decoy("image_search", "find_photo",
          "Search the internet for images matching a description.",
          ["query"], image=True, result="Image found and loaded."),
    Decoy("edit_photo", "inpaint_image",
          "Edit the current photo according to an instruction.",
          ["instruction"], image=True, result="Photo edited."),
    Decoy("enhance_photo", "redraw_image",
          "Improve the quality and resolution of the current photo.",
          ["instruction"], image=True, result="Photo enhanced."),
    Decoy("image_info", "inspect_image",
          "Get information about what is in the current image.",
          ["question"], result="The image contains the requested subject."),
    Decoy("retouch_hands", "fix_hands",
          "Retouch hands and fingers in the current photo.",
          ["instruction"], image=True, result="Hands retouched."),
    Decoy("clean_image", "fix_artifact",
          "Clean up blemishes, spots and noise in the current photo.",
          ["instruction"], image=True, result="Image cleaned."),
    Decoy("animate_clip", "generate_video",
          "Turn a description or a picture into a short animated clip.",
          ["description"], result="Clip created."),
    Decoy("compute", "calculate",
          "Compute the value of a mathematical expression.",
          ["expression"], result="Computed: 42."),
    Decoy("save_note", "remember_fact",
          "Save a note about the user for later.",
          ["note"], result="Note saved."),
    Decoy("slide_deck", "create_presentation",
          "Build a slide deck about a topic.",
          ["topic"], result="Slide deck built."),
    Decoy("paste_buffer", "read_clipboard",
          "Return the text currently in the system clipboard.",
          [], result="Clipboard: Der Hund schläft auf dem Sofa."),
]

BY_NAME = {d.name: d for d in DECOYS}


# Tools personality_node withholds from EVERY turn, whatever the message says.
# deep_research is button-only: the GUI/Telegram toggle routes those requests
# past the agent entirely, so the loop drops the schema unconditionally.
# Verified against the source below rather than trusted, because a decoy that
# shadows a withheld tool is unbeatable -- the model is offered the imitation
# and not the original, so it "steals" 100% of the time and the bench reports a
# description weakness that does not exist. That is exactly what happened on
# the first run: research_report scored a 50% steal on route_deep purely
# because deep_research was never on the table.
_ALWAYS_DROPPED = frozenset({"deep_research"})


def _assert_drop_list_current(graph_personality):
    """Fail loudly if the loop's drop set moves out from under this bench."""
    import inspect
    src = inspect.getsource(graph_personality.personality_node)
    if '_drop = {"deep_research"}' not in src:
        raise SystemExit(
            "bench/tc_canary.py: personality_node's unconditional drop set has "
            "changed. Update _ALWAYS_DROPPED to match, or the decoy steal rates "
            "will be measured against tools the model was never offered.")


def _schemas_for(graph, case):
    """Real schemas + the decoys that are legitimately on offer this turn.

    Image decoys are withheld on text-only turns because production withholds
    the real image tools there too (personality_node drops _IMAGE_TOOL_NAMES
    when there is no picture and no image intent). Offering a decoy the real
    tool would not have competed with would manufacture a steal. Decoys
    shadowing an always-dropped tool are withheld for the same reason.
    """
    live = [d for d in DECOYS
            if (not d.image or case.get("image"))
            and d.shadows not in _ALWAYS_DROPPED]
    return list(graph.TOOL_SCHEMAS) + [d.schema for d in live], live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--cat", action="append", default=None)
    ap.add_argument("--id", action="append", default=None)
    ap.add_argument("--out", default="outputs/_bench_tc_canary.json")
    args = ap.parse_args()

    graph, ctx, image_stub = setup()
    import graph_personality
    _assert_drop_list_current(graph_personality)
    cases = cases_for(args.cat, args.id)
    # A chaos case's whole point is a broken tool; decoy attribution would be
    # muddied by the recovery. Canary runs on the clean cases only.
    cases = [c for c in cases if not c.get("fail")]

    print(f"{len(DECOYS)} decoys vs {len(graph.TOOL_SCHEMAS)} real tools, "
          f"{len(cases)} cases x {args.reps} rep(s)\n")

    rows = []
    for rep in range(args.reps):
        for case in cases:
            schemas, live = _schemas_for(graph, case)
            rec = ToolRecorder(extra_stubs={d.name: d.stub for d in live})
            r = run_case(graph, ctx, case, image_stub, recorder=rec, schemas=schemas)
            r["rep"] = rep
            r["offered"] = [d.name for d in live]
            r["stolen"] = [n for n in r["calls"] if n in BY_NAME]
            rows.append(r)
            tag = ("STEAL " + ",".join(r["stolen"])) if r["stolen"] else "clean"
            print(f"[{tag:<28}] {r['id']:<24} {r['calls']}")

    _report(rows, args.out)


def _report(rows, out_path):
    # opportunities: (target = the decoy's partner was the right answer,
    #                 other  = every other case where it was still offered)
    stat = defaultdict(lambda: dict(t_opp=0, t_hit=0, o_opp=0, o_hit=0))
    for r in rows:
        for name in r["offered"]:
            d = BY_NAME[name]
            target = (r.get("expect_tool") == d.shadows)
            k = stat[name]
            hit = name in r["stolen"]
            if target:
                k["t_opp"] += 1
                k["t_hit"] += hit
            else:
                k["o_opp"] += 1
                k["o_hit"] += hit

    print("\n" + "=" * 78)
    print(f"  {'decoy':<16}{'shadows':<20}{'steal':>12}{'stray':>12}")
    print("-" * 78)

    def rate(h, o):
        return f"{h}/{o} {100*h/o:5.0f}%" if o else f"{h}/0     -"

    ranked = sorted(stat.items(),
                    key=lambda kv: -(kv[1]["t_hit"] / max(kv[1]["t_opp"], 1)))
    for name, k in ranked:
        print(f"  {name:<16}{BY_NAME[name].shadows:<20}"
              f"{rate(k['t_hit'], k['t_opp']):>12}{rate(k['o_hit'], k['o_opp']):>12}")
    total_steals = sum(1 for r in rows if r["stolen"])
    print("-" * 78)
    print(f"  runs with ANY decoy called: {total_steals}/{len(rows)} "
          f"({100*total_steals/max(len(rows),1):.0f}%)")
    print("\n  A decoy with a high steal rate names an UNDER-SPECIFIED real "
          "description:\n  the model was not choosing that tool on its prose, "
          "only on its absence of rivals.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\n  raw -> {out_path}")


if __name__ == "__main__":
    main()
