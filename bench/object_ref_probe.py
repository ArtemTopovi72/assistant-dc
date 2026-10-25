"""A named real thing drawn from words alone vs with a web photo of it as a Klein reference.
«автомат Калашникова» came out as a toy, «ВАЗ 2107» as a Fiat, the «Урал» sidecar as a
tank (debug chat 2026-10-25): Klein knows the class, not the model.

    venv/Scripts/python.exe bench/object_ref_probe.py      -> tmp/object_ref/<n>.jpg (left: words, right: + photo)
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
import search
from models import Context
from seams import vision_model

OUT = ROOT / "tmp" / "object_ref"
# (search query for the photo, the thing in words, the picture asked for)
CASES = [
    ("AK-47 rifle side view photo", "an AK-47 Kalashnikov rifle",
     "A photorealistic AK-47 Kalashnikov rifle lying on a rustic wooden table, warm side light."),
    ("AK-47 rifle side view photo", "an AK-47 Kalashnikov rifle",
     "A photorealistic soldier in camouflage at a shooting range holding an AK-47 Kalashnikov rifle."),
    ("ВАЗ 2107 фото", "a VAZ-2107 Lada car",
     "A photorealistic VAZ-2107 Lada sedan parked on a snowy city street in winter."),
    ("мотоцикл Урал с коляской фото", "a Ural motorcycle with a sidecar",
     "A photorealistic Ural motorcycle with a sidecar standing by a wooden village house at sunset."),
    ("Эйфелева башня фото", "the Eiffel Tower",
     "A photorealistic view of the Eiffel Tower at dawn seen from the river, a boat in front."),
]


def find_photo(ctx, query, thing, dest):
    look = vision_model(ctx)
    for url in search.image_search_urls(ctx, query, prefer="landscape")[:10]:
        if not search.download_image(ctx, url, dest):
            continue
        ans = look(ctx, image_paths=[("Photo:", dest)], temperature=0.0, max_tokens=5,
                   user_text=f"Is this a real photograph showing {thing} whole and clearly, as the main "
                             "subject, with no text or watermark over it? Answer yes or no.") or ""
        if ans.strip().lower().startswith("yes"):
            return dest
    return None


def render(ctx, prompt, seed, refs=()):
    up = [comfy_client._upload_image_to_comfy(r, comfy_client.COMFY_URL) for r in refs]
    wf = klein.t2i_workflow(prompt, 1344, 768, seed, refs=[u for u in up if u])
    return comfy_client._submit_and_poll(ctx, wf, timeout=600, label="klein ref probe", exclusive=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = Context(models=None, transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.no_think = True
    photos = {}
    for n, (query, thing, _) in enumerate(CASES):
        if query not in photos:
            photos[query] = find_photo(ctx, query, thing, str(OUT / f"ref_{n}.jpg"))
            print("photo", query, "->", photos[query])
    for n, (query, thing, prompt) in enumerate(CASES):
        ref = photos[query]
        seed = 1000 + n
        plain = render(ctx, prompt, seed)
        with_ref = render(ctx, prompt + f" The {thing.split(' ', 1)[1]} looks exactly like the one in "
                          "image 1: its shape, proportions and every part; nothing else of image 1 "
                          "is in the picture.", seed, [ref]) if ref else None
        tiles = [Image.open(p).convert("RGB") for p in (plain, with_ref) if p]
        sheet = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width
        sheet.save(OUT / f"{n}.jpg", quality=85)
        print(n, "plain", plain, "| ref", with_ref)


if __name__ == "__main__":
    main()
