"""Live: does the script cut keep every step of the request? Scripts in the agent's own words
for camera-led and multi-step requests go through split_script -> validate_script (no render).
A camera request «облетает по кругу и останавливается на …» lost its second step to the judge
(«repeats the circling motion», «two camera moves») and went as one take with no plan to OK
(live 2026-10-21, four requests in a row).

    venv/Scripts/python bench/video_script_steps_live.py [--repeat 2]
"""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge", "voice")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import video as V

CASES = [  # (script as the agent writes it, words of the LAST step that must survive)
    ("The camera performs a smooth, cinematic circular orbit around the boy in the yellow raincoat "
     "standing on the rainy platform, capturing the train and the station roof. As the rotation "
     "completes, the camera slows and pushes in to a close-up on the boy's face as he smiles.",
     ("face", "smil"), False),
    ("The camera circles slowly around the old station clock hanging under the roof. Then it stops "
     "and zooms in close on the clock face as the minute hand jumps forward.", ("minute hand", "jump"), False),
    ("The camera flies along the blue train from the last carriage to the locomotive, then rises "
     "above the roof of the train and stops on the departure board.", ("board",), False),
    ("The boy walks to the edge of the platform and waves at the train. The train starts moving "
     "and slowly pulls away from the station.", ("pull",), False),
    ("The camera orbits the lighthouse on the cliff at sunset, then tilts up and stops on the lamp "
     "at the top as it lights up.", ("lamp", "light"), False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    bad = 0
    # a setting plus a drift of the camera is not two steps: «the letter rests on the fabric.
    # The camera pans up» became two 5 s parts of nothing (2026-10-17)
    for script, last, one in CASES + [
            ("A folded letter rests on the red fabric of the table. The camera pans up slowly.", ("letter",), True),
            ("The old man sits on the bench in the park. The camera slowly drifts across the trees.", ("man",), True)]:
        for k in range(a.repeat):
            cut = V.split_script(script)
            judged = V.validate_script(None, V.mark_inserts(None, cut), max_parts=V.AUTO_MAX_PARTS)
            joined = " ".join(judged).lower()
            ok = (len(judged) == 1 if one else len(judged) >= 2) and any(w in joined for w in last)
            bad += not ok
            print(f"{'PASS' if ok else 'FAIL'} cut {len(cut)} -> judged {len(judged)} | last step kept: "
                  f"{any(w in joined for w in last)} | {script[:60]!r}")
            for i, p in enumerate(judged, 1):
                print(f"      {i}. {p[:150]}")
    print(f"\n{bad} failed")


if __name__ == "__main__":
    main()
