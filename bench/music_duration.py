"""What actually decides how long a Music3 song is.

The duration picker offers 30/60/120/180/240 seconds and the user does not get
them: 40 comes back severed mid-phrase, 180 comes back as 90. This measures why,
without rendering anything.

`MiniMaxMusic3TextEncode` runs the autoregressive acoustic planner and returns,
alongside the conditioning, a `seconds` output -- the length it decided on. The
latent is built from THAT number, not from what we asked for, and `max_duration`
is only a ceiling ("the model can end the song earlier", says the node's own
tooltip). So the planned seconds is the whole answer, and reading it needs no
UNet, no VAE and no diffusion -- but it is NOT cheap: the planner is
autoregressive over 25 frames per second of ceiling, at about 2.7 frames/s, so a
60s probe costs ~9 minutes and a 240s probe costs ~37. Measured, after the first
version of this file claimed a second per probe.

That price decides the design. The ceiling is held at 60s and the LYRIC is
swept, because the question worth the GPU time is whether the words -- the only
thing we control when writing the song -- move the planned length at all.

First measurement, kept: a 2-block lyric asked for 30s came back planned at
exactly 30.0. The planner ran all 751 frames and never stopped early, so at 30s
the ask is a genuine ceiling and the words wanted more than the slot.

Run: venv/Scripts/python.exe bench/music_duration.py [--ceiling 60] [--blocks 1,2,4,8]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLIP_NAME = "minimax_music3_text_encoder_bf16.safetensors"

CAPTION = ("Global Metadata: indie pop, 108 BPM, key of C major, warm and bright.\n"
           "Vocal Details: single female lead, close and clear, light double-track "
           "on the chorus.\n"
           "Arrangement: acoustic guitar and soft drums, bass entering on verse two, "
           "strings under the bridge.")

_VERSE = ("[verse]\n"
          "Morning on the kitchen floor, the kettle starts to sing\n"
          "You were counting all the ways that quiet is a kind of spring\n"
          "Every cup we ever filled is standing in a row\n"
          "And nobody has to say the thing that both of us already know\n")
_CHORUS = ("[chorus]\n"
           "So hold the light a little longer, hold it while it lasts\n"
           "We are learning how to carry what we could not carry past\n"
           "Hold the light a little longer, let the morning stay\n"
           "We are learning how to keep it when it wants to walk away\n")
_OUTRO = ("[outro]\n"
          "Hold the light a little longer, let the morning stay\n")


def lyric(n_blocks: int) -> str:
    """A song of `n_blocks` verse+chorus pairs, always closed with an outro."""
    body = "".join(_VERSE + _CHORUS for _ in range(n_blocks))
    return "[intro]\n\n" + body + _OUTRO


def graph(lyrics: str, max_duration: float, seed: int = 7) -> dict:
    """CLIPLoader -> MiniMaxMusic3TextEncode -> PreviewAny(seconds). Nothing else."""
    return {
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"}},
        "4": {"class_type": "MiniMaxMusic3TextEncode",
              "inputs": {"clip": ["2", 0], "caption": CAPTION, "lyrics": lyrics,
                         "seed": seed, "max_duration": float(max_duration),
                         "cfg_scale": 1.7, "top_k": 50}},
        "10": {"class_type": "PreviewAny", "inputs": {"source": ["4", 1]}},
    }


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def planned_seconds(base: str, lyrics: str, max_duration: float, seed: int = 7,
                    timeout: int = 3600):
    """Ask the planner how long it intends the song to be. None if it failed."""
    try:
        pid = _post(base + "/prompt", {"prompt": graph(lyrics, max_duration, seed)})["prompt_id"]
    except urllib.error.HTTPError as exc:
        print("  submit rejected: %s" % exc.read().decode("utf-8", "replace")[:400], flush=True)
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(base + "/history/" + pid, timeout=30) as r:
            hist = json.loads(r.read().decode("utf-8"))
        if pid in hist:
            outs = hist[pid].get("outputs", {})
            node = outs.get("10") or {}
            text = node.get("text") or node.get("string") or []
            if text:
                try:
                    return float(str(text[0]).strip())
                except ValueError:
                    print("  unreadable seconds: %r" % (text,), flush=True)
                    return None
            status = hist[pid].get("status", {})
            print("  no seconds in outputs (%s)" % status.get("status_str"), flush=True)
            return None
        time.sleep(1.0)
    print("  timed out", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8000")
    ap.add_argument("--ceiling", type=float, default=60.0)
    ap.add_argument("--blocks", default="1,2,4,8")
    args = ap.parse_args()
    base = args.comfy.rstrip("/")
    ceiling = args.ceiling

    print("ceiling fixed at %gs, lyric length swept" % ceiling, flush=True)
    print("   %-8s %-8s %-8s %-10s %s"
          % ("blocks", "lines", "chars", "planned", "verdict"), flush=True)
    for n in [int(b) for b in args.blocks.split(",") if b.strip()]:
        ly = lyric(n)
        t0 = time.time()
        got = planned_seconds(base, ly, ceiling)
        if got is None:
            continue
        lines_n = len([l for l in ly.splitlines()
                       if l.strip() and not l.strip().startswith("[")])
        verdict = ("CAPPED - the words wanted more"
                   if got >= ceiling - 0.5 else
                   "stopped %.0fs early - the words ran out" % (ceiling - got))
        print("   %-8d %-8d %-8d %-10s %s   [%.0fs]"
              % (n, lines_n, len(ly), "%.1fs" % got, verdict, time.time() - t0),
              flush=True)


if __name__ == "__main__":
    main()
