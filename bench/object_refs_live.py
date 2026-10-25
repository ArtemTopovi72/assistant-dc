"""object_refs on the house model: which requests name a specific real thing to fetch a photo
of (--read, text only), and the drawings with and without that photo (default; same seed,
left: words, right: + photo -> tmp/object_refs/<n>.jpg).

    venv/Scripts/python.exe bench/object_refs_live.py --read
    venv/Scripts/python.exe bench/object_refs_live.py
"""
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "bench"]
for d in ("", "agent", "imaging", "media", "bot", "core", "voice", "knowledge", "research", "services"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.pop("F5_TEST_RUN", None)

import object_refs
from models import Context

# (request, the kinds it names to look up)
READ = [
    ("ВАЗ 2107 на заснеженной улице", ["object"]),
    ("мотоцикл Урал с коляской у деревенского дома", ["object"]),
    ("красная гитара Gibson Les Paul на подставке", ["object"]),
    ("трактор Кировец К-700 в поле", ["object"]),
    ("самолёт Ту-154 на взлётной полосе", ["object"]),
    ("iPhone 15 Pro на столе рядом с кофе", ["object"]),
    ("Ленин в стиле Думгая в аду дерётся с демонами", ["person", "character"]),
    ("Гагарин в скафандре машет рукой", ["person"]),
    ("кот в костюме Бэтмена на крыше", ["character"]),
    ("Шрек пьёт чай на кухне", ["character"]),
    ("машина едет по ночному городу", []),
    ("кот спит на подоконнике", []),
    ("Эйфелева башня на рассвете", []),
    ("девушка в красном платье танцует", []),
    ("гитара у костра", []),
    ("рыцарь сражается с драконом", []),
]

DRAW = [
    "A photorealistic VAZ-2107 Lada sedan parked on a snowy city street in winter.",
    "A photorealistic red VAZ-2107 Lada sedan seen from behind, driving away down a country road in summer.",
    "A photorealistic Ural motorcycle with a sidecar standing by a wooden village house at sunset.",
    "A photorealistic Kirovets K-700 tractor ploughing a field under a cloudy sky.",
    "A photorealistic Gibson Les Paul guitar on a stand in a small music room.",
    "A cinematic, hyper-realistic action shot of Vladimir Lenin reimagined as a heavy-armored Doomguy-style "
    "warrior, waist-deep in glowing orange lava in hell. He grins fiercely, his eyes glow with ruby light and a "
    "glowing red star shines on his forehead. With one armored fist he has just crushed a demon's skull; his "
    "other hand squeezes the neck of a second demon.",
]


def ctx_():
    c = Context(models=None, transcription_cache={}, cache_file=None,
                asr_lock=threading.Lock(), tts_lock=threading.Lock())
    c.no_think = True
    return c


def read():
    ctx, ok = ctx_(), 0
    for text, want in READ:
        got = object_refs.named_things(ctx, text)
        kinds = sorted(t["kind"] for t in got)
        ok += kinds == sorted(want)
        print(("PASS " if kinds == sorted(want) else "FAIL ") + f"{text!r} -> {got}")
    print(f"{ok}/{len(READ)}")


def draw(only=None):
    import comfy_client
    import klein
    from PIL import Image
    out = ROOT / "tmp" / "object_refs"
    out.mkdir(parents=True, exist_ok=True)
    ctx = ctx_()
    for n, prompt in enumerate(DRAW):
        if only is not None and n != only:
            continue
        things = object_refs.find(ctx, prompt)
        print(n, "things:", things)
        paths = []
        for refs in ([], things):
            up = [comfy_client._upload_image_to_comfy(p, comfy_client.COMFY_URL) for _, p, _ in refs]
            wf = klein.t2i_workflow(prompt + object_refs.prompt_note(refs), 1344, 768, 2000 + n, refs=up)
            paths.append(comfy_client._submit_and_poll(ctx, wf, timeout=600, label="objref bench", exclusive=True))
            if not refs:
                if not things:
                    break
        tiles = [Image.open(p).convert("RGB") for p in paths if p]
        tiles += [Image.open(p).convert("RGB").resize((int(768 * Image.open(p).width / Image.open(p).height), 768))
                  for _, p, _ in things]
        sheet = Image.new("RGB", (sum(t.width for t in tiles), 768))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width
        sheet.save(out / f"{n}.jpg", quality=85)
        print(n, "done", paths)


if __name__ == "__main__":
    if "--read" in sys.argv:
        read()
    else:
        draw(int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None)
