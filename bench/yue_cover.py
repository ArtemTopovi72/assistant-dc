"""A cover through YuE2's own cover path: the recording's melody (SheetSage2 -> ABC score) steers the song, cot=melody.
    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/yue_cover.py song.mp4 lyrics.txt "<style>" [--name x] [--seed 7]
Stop the desktop app first (the render wants the card)."""
import json, os, subprocess, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import music

src, lyr_file, style = sys.argv[1], sys.argv[2], sys.argv[3]
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
name, seed = arg("--name", "yue"), int(arg("--seed", 7))
OUT = ROOT / "outputs/yue_cover"; OUT.mkdir(parents=True, exist_ok=True)
CPP = ROOT / "models_ext/wsl_src/yue2.cpp-master/build"
GG = ROOT / "models_ext/wsl_src/gguf"
env = {**os.environ, "PATH": os.pathsep.join([str(CPP), os.path.join(os.environ.get("CUDA_PATH", ""), "bin"), os.environ["PATH"]])}
ref = OUT / "ref.wav"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "2", "-ar", "44100", str(ref)], check=True)
abc = OUT / "score_melody.abc"
subprocess.run([str(CPP / "yue-transcribe.exe"), "--model", str(GG / "SheetSage2-Q8_0.gguf"), "--audio", str(ref),
                "--out", str(abc), "--melody-only"], env=env, check=True, capture_output=True)
out = OUT / f"{name}.mp3"
job = {"lyrics": music.yue2_lyrics(Path(lyr_file).read_text(encoding="utf-8")), "style": style, "abc": abc.read_text(encoding="utf-8"),
       "cot": "melody", "lm_seed": seed, "steps": music.YUE2_ODE_STEPS}
jf = OUT / f"{name}.json"; jf.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
t0 = time.time()
r = subprocess.run([str(CPP / "yue-synth.exe"), "--model", str(GG / "YuE2-3B-BF16.gguf"), "--vae", str(GG / "YuE2-Vae-F32.gguf"),
                    "--request", str(jf), "--out", str(out)], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
print("rc", r.returncode, f"{time.time() - t0:.0f}s", out.exists())
print((r.stdout + r.stderr)[-500:])
music._master(str(out), fade=False)
