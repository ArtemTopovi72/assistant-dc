"""How to write «рюкзачком» so H3 says рюкзачкОм: run 19's last talking-head part (its own H3
prompt, the overseer still as first = last frame), the word spelled several ways, two seeds each.
H3 ignores U+0301 and CAPS (A/B 2026-10-06); this tries spellings and a prose cue.
    venv/Scripts/python.exe bench/video_stress_spelling.py tmp/final/stress_prompt.txt tmp/overseer.jpg [variant ...]"""
import logging
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging", "media", "bot", "core"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.chdir(ROOT)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import models  # noqa: E402
import video  # noqa: E402

prompt_path, still = sys.argv[1:3]
base = open(prompt_path, encoding="utf-8").read()
word = "рюкзачко́м"
assert word in base, "the prompt has no stressed рюкзачком"
cue = (" He says the word «рюкзачком» with the stress on its last syllable, «ком»: "
       "рюкзач-КОМ, never рюкЗАчком.")
variants = {
    "acute": base,
    "double": base.replace(word, "рюкзачкоом"),
    "hyphen": base.replace(word, "рюкзач-ко́м"),
    "cue": base.replace("</d>", "</d>" + cue, 1),
    # the cue above gave the cleanest stress (owner, 2222) but the word was said twice: it quotes
    # the word, and H3 speaks quoted Russian. These name it without a speakable form.
    "cue_pos": base.replace("</d>", "</d> In his line the fifth word is stressed on its final "
                            "syllable, never on its second.", 1),
    # by position: no repeat, but the stress came back wrong on both takes (owner, 2026-10-21)
    "cue_gloss": base.replace("</d>", "</d> In his line the word that means 'with the backpack' is "
                              "stressed on its last syllable, never on its second.", 1),
    "cue_latin": base.replace("</d>", "</d> In his line the word ryukzachkom is stressed on its last "
                              "syllable: ryukzach-KOM, never ryuk-ZACH-kom.", 1),
    "cue_once":base.replace("</d>", "</d>" + cue + " He says the line once, exactly as written; "
                             "this note about stress is never spoken.", 1),
}
ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
out = os.path.join(ROOT, "tmp", "stress_spelling")
os.makedirs(out, exist_ok=True)
for name in sys.argv[3:] or list(variants):
    for seed in (1111, 2222):
        t0 = time.time()
        r = video.generate_video(ctx, variants[name], images=[still], keyframe=True, prepared=True,
                                 seconds=5.17, seed=seed)
        dst = os.path.join(out, f"{name}_{seed}.mp4")
        if r.get("path"):
            os.replace(r["path"], dst)
        print("RESULT", name, seed, r.get("status"), dst if r.get("path") else r.get("reason"),
              f"{time.time() - t0:.0f}s", flush=True)
