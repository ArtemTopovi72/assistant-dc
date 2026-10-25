"""Does the H3 prompt writer cut a part into shots the script never asked for? A chain's
part 2 came back «[Shot 2] At 00:03.500, the camera cuts to a close-up of the golden
retriever's face» for a script with no cut in it, and the clip jumped mid-part
(debug chat 2026-10-25). Real split/bridge/prepare_prompt on real LM Studio, no render.

    venv/Scripts/python.exe bench/video_cut_probe.py [runs]
"""
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "bench"]
for d in ("", "agent", "imaging", "media", "bot", "core", "voice", "knowledge", "research", "services", "sandbox"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.pop("F5_TEST_RUN", None)

import video as V
from models import Context

# (script, a cut is asked for)
CASES = [
    ("A golden retriever runs energetically along a sandy beach, its paws kicking up sand; the "
     "camera follows closely at a low angle, tracking the dog's movement. The dog then leaps into "
     "a crashing ocean wave, splashing water everywhere, and barks excitedly as it emerges from "
     "the foam; sound of waves and wind.", False),
    ("A cat walks along a wooden windowsill, looking out at the rain. Then the cat jumps down "
     "onto a soft rug and has a long, satisfying stretch.", False),
    ("A girl flies a red kite on a green hill. The kite climbs high into the sky, and she "
     "laughs and runs after it, holding the string.", False),
    ("A cook cracks an egg into a hot pan and it sizzles. He sprinkles salt over it, then "
     "flips the egg with a spatula and slides it onto a plate.", False),
    ("A boy throws a ball across the yard and his dog catches it in the air. Then a close-up "
     "of the dog's happy face with the ball in its mouth.", True),
    ("A wide shot of a small town square at dawn. Cut to a close-up of the clock on the tower "
     "as its hands reach six and the bell rings.", True),
]

# continued parts handed to prepare_prompt as they were live: (part, scene so far, a cut is asked for)
PARTS = [
    ("The camera follows closely at a low angle behind the dog as it leaps into a crashing ocean "
     "wave and emerges from the foam; \"bark!\"",
     "A golden retriever runs along a sandy beach, kicking up sand; sound of rhythmic sea waves, "
     "wind whistling, and a joyful bark.", False),
    ("The cat jumps down from the sill onto a soft rug and has a long, satisfying stretch.",
     "A cat walks along a wooden windowsill, looking out at the rain.", False),
    ("The girl lets go of the string and the kite flies away over the hill; she waves after it.",
     "A girl flies a red kite on a green hill, laughing.", False),
    ("The dog drops the ball at the boy's feet. Then cut to a close-up of the boy's grinning face.",
     "A boy throws a ball across the yard and his dog catches it.", True),
]

SHOT2 = re.compile(r"\[Shot [2-9]\]")


def parts_of(ctx, script):
    parts = V.split_script(script)
    if len(parts) > 1:
        parts = V.bridge_script(ctx, parts)
    out = [V.prepare_prompt(ctx, parts[0])[0]]
    for i, p in enumerate(parts[1:], start=1):
        out.append(V.prepare_prompt(ctx, V.CONTINUE_CTX_PREFIX + V.with_scene(p, " ".join(parts[:i])))[0])
    return out


def main(runs):
    ctx = Context(models=None, transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.no_think = True
    ctx.tts_disabled = True
    bad = total = 0
    for part, scene, cut_asked in PARTS:
        for r in range(runs):
            prompt = V.prepare_prompt(ctx, V.CONTINUE_CTX_PREFIX + V.with_scene(part, scene), seconds=5.2)[0]
            cut = bool(SHOT2.search(prompt))
            if cut_asked:
                print(f"{'info':5} cut={cut} continued: {part[:50]!r}")
                continue
            total += 1
            bad += cut
            print(f"{'FAIL' if cut else 'PASS':5} continued: {part[:50]!r}")
            if cut:
                print("      " + prompt[:700].replace("\n", " "))
    if "--parts" in sys.argv:
        print(f"unasked cuts: {bad}/{total} parts")
        return
    for script, cut_asked in CASES:
        for r in range(runs):
            for n, prompt in enumerate(parts_of(ctx, script), 1):
                cut = bool(SHOT2.search(prompt))
                if cut_asked:
                    print(f"{'info':5} cut={cut} part {n}: {script[:50]!r}")
                    continue
                total += 1
                bad += cut
                print(f"{'FAIL' if cut else 'PASS':5} part {n}: {script[:50]!r}")
                if cut:
                    print("      " + prompt[:700].replace("\n", " "))
    print(f"unasked cuts: {bad}/{total} parts")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2)
