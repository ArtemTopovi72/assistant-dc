"""Which WORDING of the scene sentence actually holds the frame together.

bench/draw_collage_probe.py established the cause: the `two_of_the_same` layout
renders as a 2x2 collage when `high_level_description` is empty, and renders as
one coherent room when a hand-written sentence is supplied. A rich background
does not help.

Then the synthesized floor shipped in ideogram_layout.synth_high_level() FAILED
to reproduce that cure -- twice, measured, at the same seed. The renders are
byte-deterministic at a fixed seed (the hand-written variant reproduced exactly
across two runs), so the difference is the sentence and nothing else:

    works : "Одна фотография одной комнаты: чёрный кот на полу, рыжий кот спит
             в кресле, деревянный стол — все в одном кадре."
    fails : "Одно изображение одной сцены: комната. …"          (collage)
    fails : "Одна фотография одной сцены: комната. … на одном снимке."  (collage)

Three suspects separate those: the abstract noun "сцена", the full stop that
splits the line into two sentences, and the trailing clause. Guessing costs ~90s
a render, so this sweeps the candidates in one run instead, with the known-good
sentence included as a CONTROL -- if the control does not come back as one room,
the harness is wrong and no conclusion may be drawn from the rest.

Run: venv/Scripts/python.exe bench/draw_scene_floor_probe.py --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ideogram as IG

SEED = 12345

BG = "комната"
ELS = ["чёрный кот сидит на полу", "рыжий кот спит на кресле", "деревянный стол"]

LAYOUT = {
    "background": BG,
    "elements": [
        {"desc": ELS[0], "x": 0.10, "y": 0.60, "w": 0.22, "h": 0.28},
        {"desc": ELS[1], "x": 0.60, "y": 0.55, "w": 0.26, "h": 0.30},
        {"desc": ELS[2], "x": 0.30, "y": 0.20, "w": 0.34, "h": 0.28},
    ],
}

_joined = ", ".join(ELS)

CANDIDATES = {
    # The sentence that is known to cure it. If this one collages, stop.
    "control": ("Одна фотография одной комнаты: чёрный кот на полу, рыжий кот "
                "спит в кресле, деревянный стол — все в одном кадре."),
    # What ships today, and is known to fail. Kept so the run is self-evidencing.
    "shipped": ("Одна фотография одной сцены: %s. %s — всё это в одном кадре, "
                "на одном снимке." % (BG, _joined)),
    # Control's shape, built mechanically: no abstract "сцена", no full stop
    # inside the line, background used as a bare noun phrase.
    "no_scene_word": "Одна фотография: %s — %s, все в одном кадре." % (BG, _joined),
    # As above, but leading with the frame rather than the medium.
    "one_frame_first": ("Один кадр, одна комната: %s. Все они вместе на одной "
                        "фотографии." % _joined),
    # The shipped sentence with ONLY the full stop removed, to isolate it.
    "no_full_stop": ("Одна фотография одной сцены — %s: %s — всё это в одном "
                     "кадре." % (BG, _joined)),

    # -- second round. The first three all collaged, so none of "сцена", the
    # full stop or the trailing clause was the operative difference. What is
    # left is that the control says the photograph is OF THE ROOM, and that it
    # PARAPHRASES the elements instead of quoting their descs verbatim.

    # The control's skeleton, but quoting the element descs exactly as the
    # layout wrote them. If this works, the template is what matters and the
    # floor can be built mechanically; if it collages, the paraphrase is doing
    # the work and a mechanical floor cannot reproduce it.
    "control_verbatim": ("Одна фотография одной комнаты: %s — все в одном "
                         "кадре." % _joined),
    # Says the photo is OF a place without having to inflect the background:
    # a quoted noun after a genitive head word.
    "photo_of_place": ("Одна фотография места «%s»: %s — все в одном кадре."
                       % (BG, _joined)),
    # The control with ONE word changed, to check the preposition is not the
    # hidden variable (the layout says "на кресле", the control "в кресле").
    "control_na_kresle": ("Одна фотография одной комнаты: чёрный кот на полу, "
                          "рыжий кот спит на кресле, деревянный стол — все в "
                          "одном кадре."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--only", default="", help="comma-separated candidate names")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = [w.strip() for w in args.only.split(",") if w.strip()] or list(CANDIDATES)
    for name in wanted:
        sentence = CANDIDATES[name]
        layout = dict(LAYOUT, high_level_description=sentence)
        caption = IG.layout_to_caption(IG.normalize_layout(layout))
        (out / (name + ".caption.json")).write_text(
            json.dumps(caption, ensure_ascii=False, indent=2), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            # ctx=None: nothing here may call the LLM; the card is the
            # renderer's for the duration of the probe.
            path = IG.generate(None, "", width=args.width, height=args.height,
                               caption=caption, seed=SEED)
            why = path or "the renderer returned nothing"
        except Exception as exc:
            path, why = None, "%s: %s" % (type(exc).__name__, exc)
        if path:
            dest = out / (name + Path(path).suffix)
            dest.write_bytes(Path(path).read_bytes())
            why = str(dest)
        print("[draw] %-16s %6.1fs  %s" % (name, time.perf_counter() - t0, why),
              flush=True)


if __name__ == "__main__":
    main()
