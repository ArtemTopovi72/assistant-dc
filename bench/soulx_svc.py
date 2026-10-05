"""SoulX-Singer-SVC: sing `source` in the timbre of `prompt`. Runs in the main venv (F0 via pyworld), model in venv_soulx on E:.
    from soulx_svc import convert; convert(source.wav, prompt.wav, out.wav)"""
import subprocess, tempfile
from pathlib import Path
import librosa, numpy as np, pyworld as pw, soundfile as sf
ROOT = Path(__file__).resolve().parent.parent
SX = ROOT / "models_ext/soulx_singer"
PY = Path("E:/venvs/venv_soulx/Scripts/python.exe")
SR, HOP = 24000, 480


def _f0(wav24: Path, npy: Path):
    y, _ = sf.read(wav24, dtype="float64")
    f, _ = pw.harvest(y, SR, f0_floor=70, f0_ceil=800, frame_period=HOP / SR * 1000)
    np.save(npy, f.astype(np.float32)[: (len(y) // HOP) + 1])


def convert(source, prompt, out, shift=0, steps=32):
    d = Path(tempfile.mkdtemp(prefix="sx_", dir="E:/venvs/tmp"))
    for k, p in (("src", source), ("prm", prompt)):
        y, _ = librosa.load(p, sr=SR, mono=True)
        sf.write(d / f"{k}.wav", y, SR)
        _f0(d / f"{k}.wav", d / f"{k}_f0.npy")
    r = subprocess.run([str(PY), "-m", "cli.inference_svc", "--device", "cuda", "--model_path", "pretrained_models/SoulX-Singer/model-svc.pt",
                        "--config", "soulxsinger/config/soulxsinger.yaml", "--prompt_wav_path", str(d / "prm.wav"), "--target_wav_path", str(d / "src.wav"),
                        "--prompt_f0_path", str(d / "prm_f0.npy"), "--target_f0_path", str(d / "src_f0.npy"), "--save_dir", str(d / "o"),
                        "--auto_shift", "--pitch_shift", str(shift), "--n_steps", str(steps), "--fp16"],
                       cwd=SX, env={**__import__("os").environ, "PYTHONPATH": str(SX)}, capture_output=True, text=True, encoding="utf-8", errors="replace")
    g = d / "o/generated.wav"
    if not g.exists():
        raise RuntimeError(r.stderr[-1500:])
    Path(out).write_bytes(g.read_bytes())


if __name__ == "__main__":
    import sys
    convert(*sys.argv[1:4]); print("ok", sys.argv[3])
