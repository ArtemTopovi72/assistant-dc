"""Transcribe clip audio (Russian, GigaAM on CPU): bench/asr_clip.py a.mp4 [b.mp4 ...]"""
import os, subprocess, sys, tempfile
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import onnx_asr
m = onnx_asr.load_model("gigaam-v3-e2e-rnnt")
for p in sys.argv[1:]:
    wav = os.path.join(tempfile.gettempdir(), "_asr_clip.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-vn", "-ac", "1", "-ar", "16000", wav])
    print(p, "->", m.recognize(wav))
