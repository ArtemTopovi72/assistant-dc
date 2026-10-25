"""A web photo as inspiration, not pixels: the vision model describes the found thing in words
and Klein draws from those words, vs the photo as a reference latent. As a latent the photo
dictated the frame -- Lenin stared into the camera in a centred portrait, the «Урал» kept the
photo's angle; the owner: «нам не надо пиксели из поиска, нам ими надо вдохновиться» (2026-10-26).

    venv/Scripts/python.exe bench/inspire_probe.py [n ...]   -> tmp/inspire/<n>.jpg
    columns: words only | photo as latent | words from the photo | the photo
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

from PIL import Image
import comfy_client
import klein
import object_refs as R
from models import Context
from seams import vision_model

CASES = [
    "A photorealistic VAZ-2107 Lada sedan drifting sideways on a snowy city street at night, headlights on.",
    "A photorealistic Ural motorcycle with a sidecar racing down a dusty country road, a dog in the sidecar, seen from a low front angle.",
    "A photorealistic Kirovets K-700 tractor pulling a plough across a muddy field at sunset, seen from behind.",
    "A cinematic, hyper-realistic action shot of Vladimir Lenin reimagined as a heavy-armored Doomguy-style "
    "warrior, waist-deep in glowing orange lava in hell. He grins fiercely, his eyes glow with ruby light and a "
    "glowing red star shines on his forehead. With one armored fist he has just crushed a demon's skull; his "
    "other hand squeezes the neck of a second demon.",
    "Yuri Gagarin laughing in a Soviet kitchen in the 1960s, pouring tea for friends, warm evening light.",
]

DESCRIBE = {
    "person": "This is a photo of {thing}. Describe the FACE and HEAD for an artist who must draw {thing} "
              "recognisably from any angle, in any light, with any expression: head shape, hair or baldness "
              "and hairline, facial hair (its exact shape, length and colour), brow, eyes, nose, cheekbones, "
              "mouth, ears, apparent age. 50-80 words, visual facts only, no clothes, no pose, no background.",
    "object": "This is a photo of {thing}. Describe it for an artist who must draw it recognisably from any "
              "angle: overall shape and proportions, then each distinctive part and where it sits (front, "
              "sides, rear, top), materials. 60-90 words, visual facts only, no background, no lighting.",
    "character": "This is a picture of {thing}. Describe it for an artist who must draw it recognisably on "
                 "someone else, from any angle and in any pose: its overall shape, each distinctive part and "
                 "where it sits, materials and colours. 60-90 words, visual facts only, no pose, no background.",
}


def describe(ctx, thing, path, kind):
    return " ".join((vision_model(ctx)(ctx, image_paths=[("Photo:", path)], temperature=0.1, max_tokens=220,
                                       user_text=DESCRIBE[kind].format(thing=thing)) or "").split())


def render(ctx, prompt, seed, refs=()):
    up = [comfy_client._upload_image_to_comfy(p, comfy_client.COMFY_URL) for p in refs]
    wf = klein.t2i_workflow(prompt, 1344, 768, seed, refs=up)
    return comfy_client._submit_and_poll(ctx, wf, timeout=600, label="inspire probe", exclusive=True)


def main(only):
    out = ROOT / "tmp" / "inspire"
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(models=None, transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.no_think = True
    for n, prompt in enumerate(CASES):
        if only and n not in only:
            continue
        things = R.find(ctx, prompt)
        words = " ".join(f"{t}: {describe(ctx, t, p, k)}" for t, p, k in things)
        print(n, things, "\n   WORDS:", words)
        seed = 7000 + n
        takes = [render(ctx, prompt, seed),
                 render(ctx, prompt + R.prompt_note(things), seed, [p for _, p, _ in things]) if things else None,
                 render(ctx, prompt + " " + words, seed) if words else None]
        tiles = [Image.open(p).convert("RGB").resize((672, 384)) for p in takes if p]
        tiles += [Image.open(p).convert("RGB").resize((int(384 * Image.open(p).width / Image.open(p).height), 384))
                  for _, p, _ in things]
        sheet = Image.new("RGB", (sum(t.width for t in tiles), 384))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width
        sheet.save(out / f"{n}.jpg", quality=85)
        print(n, "done", takes)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
