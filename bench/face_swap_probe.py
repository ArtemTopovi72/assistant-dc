"""A named person: the scene drawn free first, then only the face edited in from a web photo.

Given to the scene render as a reference latent, the face photo set the frame: Lenin stared into
the camera in a centred portrait, the fist fight gone; the owner: «башка должна смотреть куда надо,
цветокор по сюжету, фото брать для вдохновения» (2026-10-26).

    venv/Scripts/python.exe bench/face_swap_probe.py [n ...]   -> tmp/faceswap/<n>.jpg
    columns: scene (references but the face) | + face edited in | face photo
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
import image as I
import klein
import object_refs as R
from models import Context

CASES = [
    "A cinematic, hyper-realistic action shot of Vladimir Lenin reimagined as a heavy-armored Doomguy-style "
    "warrior, waist-deep in glowing orange lava in hell. He grins fiercely, his eyes glow with ruby light and a "
    "glowing red star shines on his forehead. With one armored fist he has just crushed a demon's skull; his "
    "other hand squeezes the neck of a second demon.",
    "Yuri Gagarin laughing in a Soviet kitchen in the 1960s, pouring tea for friends, warm evening light.",
    "Vladimir Lenin riding a galloping white horse across a snowy field, waving his cap, seen from the side.",
]

SWAP = ("The face and head of {who} in this picture become the face of {who} from image 2 -- his "
        "features, head shape, hairline and facial hair -- while his head angle, gaze, expression, everything "
        "on his face this picture adds (glow, marks, paint) and this scene's light and colours stay as they are.")


def main(only):
    out = ROOT / "tmp" / "faceswap"
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(models=None, transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.no_think = True
    for n, prompt in enumerate(CASES):
        if only and n not in only:
            continue
        things = R.find(ctx, prompt)
        print(n, things)
        rest = [t for t in things if t[2] != "person"]
        faces = [t for t in things if t[2] == "person"]
        up = [comfy_client._upload_image_to_comfy(p, comfy_client.COMFY_URL) for _, p, _ in rest]
        scene = comfy_client._submit_and_poll(
            ctx, klein.t2i_workflow(prompt + R.prompt_note(rest), 1344, 768, 9000 + n, refs=up),
            timeout=600, label="faceswap scene", exclusive=True)
        swapped = scene
        for who, face, _ in faces:
            swapped = I._edit_image_with_firered_once(ctx, swapped, SWAP.format(who=who), seed=9100 + n,
                                                      reference_paths=[face]) or swapped
        tiles = [Image.open(p).convert("RGB").resize((672, 384)) for p in (scene, swapped)]
        tiles += [Image.open(p).convert("RGB").resize((int(384 * Image.open(p).width / Image.open(p).height), 384))
                  for _, p, _ in faces]
        sheet = Image.new("RGB", (sum(t.width for t in tiles), 384))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width
        sheet.save(out / f"{n}.jpg", quality=85)
        print(n, "done", scene, swapped)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
