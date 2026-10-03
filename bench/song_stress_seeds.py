"""Acute accent did NOT hold in a full song (гнИлое again, 2026-09-28). One-seed
A/B was luck. 3 seeds x 3 spellings of the same two lines, judged by ear."""
import os, shutil, sys, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import music
music.stress_lyrics = lambda t, *a, **k: t         # send each spelling as-is
STYLE = "Hard rock / metal, 145 BPM, E minor, aggressive, angry female vocal"
BASE = "[chorus]\nНикто ни хера не делает, лишь хамят в лицо,\nСобрали в поликлинике сплошное {w} яйцо!"
VARIANTS = {"acute": "гнило́е", "long": "гнилоое", "phon": "гнилоэ"}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_stress_seeds")
os.makedirs(out, exist_ok=True)
ctx = types.SimpleNamespace(set_stage=lambda *a, **k: None, cancel_event=threading.Event())
for seed in (11, 22, 33):
    for name, w in VARIANTS.items():
        f = music._generate_yue2(ctx, BASE.format(w=w), STYLE, seed)
        shutil.copy(f, os.path.join(out, f"s{seed}_{name}.mp3"))
        print("->", seed, name, flush=True)
