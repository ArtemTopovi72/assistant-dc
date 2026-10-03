"""Is the song slow because of the model, or because the LLM is holding VRAM?

The GUI label promises ~40 s for a 30-second song on the `fast` preset. A real
run today took 273 s. The card had LM Studio resident on it (20.5 GB of 24.5),
so the question is whether Music3 is streaming its weights.

Same lyrics, same style, same seed, same preset, twice: once with the chat model
resident, once with the card to ourselves. Anything else on the GPU invalidates
both numbers, so this refuses to run if ComfyUI is already busy.

    venv/Scripts/python.exe bench/music_vram_ab.py [--duration 30]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import comfy_client
import lora_training as LT
import music as M

SEED = 424242
LYRICS = M.normalize_lyrics(
    "[verse]\nСтепан идёт по снегу, ветер бьёт в лицо\n"
    "Фонарь горит над дверью, и ждёт его крыльцо\n"
    "[chorus]\nДомой, домой, дорога коротка\n"
    "Домой, домой, и ночь уже близка\n")
STYLE = ("русский рок, мужской вокал, средний темп, гитара и барабаны, "
         "живое звучание, тёплый микс")


def vram():
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def run(label, duration):
    print("\n=== %s | %d MiB held before the render ===" % (label, vram()))
    t0 = time.time()
    path = M.generate_music(None, LYRICS, STYLE, duration_s=duration,
                            seed=SEED, preset="fast")
    dt = time.time() - t0
    print("%s: %.1f s -> %s" % (label, dt, path))
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30)
    args = ap.parse_args()

    busy = comfy_client.gpu_holder()
    if busy:
        print("REFUSING: ComfyUI is busy with %r -- a shared card makes both "
              "numbers meaningless." % (busy,))
        return 2

    llm = LT.loaded_llm_id()
    print("chat model resident: %r" % (llm or "none"))

    with_llm = run("A: chat model resident", args.duration)

    LT.free_gpu(log=print)
    time.sleep(5)
    without = run("B: card to ourselves", args.duration)

    if llm:
        LT.reload_llm(llm, log=print)

    print("\n---------------------------------------------")
    print("with the LLM resident : %6.1f s" % with_llm)
    print("with the card free    : %6.1f s" % without)
    if without > 0:
        print("difference            : %6.1f s  (%.2fx)"
              % (with_llm - without, with_llm / without))
    print("GUI label promises ~40 s for 30 s of audio on this preset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
