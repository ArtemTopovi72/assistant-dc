"""
VAD test using real speech WAV files injected into VadListener queue.
No microphone needed.

Run: .\\venv\\Scripts\\python.exe tests\\test_vad_synthetic.py
"""
import os, sys, time, threading
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, ".")

FRAME = 512
FS    = 16000

# Real speech: the sample f5_tts ships, wherever the environment lives.
import f5_tts
WAV_FILES = [os.path.join(list(f5_tts.__path__)[0], "infer", "examples", "basic", "basic_ref_en.wav")]

def load_speech(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != FS:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=FS)
    return audio.astype(np.float32)

def audio_to_frames(audio):
    # Pad to multiple of FRAME
    pad = (FRAME - len(audio) % FRAME) % FRAME
    audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])
    return [audio[i:i+FRAME] for i in range(0, len(audio), FRAME)]

# ── setup ──────────────────────────────────────────────────────────────────────
print("Loading Silero VAD model...")
from audio import VadListener, _load_silero_vad_model

captured = []

def on_utterance(audio):
    dur = len(audio) / FS
    rms = float(np.sqrt(np.mean(audio**2)))
    captured.append({"dur": dur, "rms": rms})
    print(f"  >>> utterance {len(captured)}: {dur:.2f}s  RMS={rms:.4f}")

listener = VadListener(on_utterance=on_utterance)
listener._model = _load_silero_vad_model()
listener._thread = threading.Thread(target=listener._run, name="vad-test", daemon=True)
listener._thread.start()

# ── inject: silence + speech + silence for each WAV ───────────────────────────
silence = np.zeros(FRAME, dtype=np.float32)

for wav_path in WAV_FILES:
    try:
        speech = load_speech(wav_path)
        break
    except Exception as e:
        print(f"  skip {wav_path}: {e}")
else:
    print("No WAV file found"); sys.exit(1)

print(f"\nUsing: {wav_path}  ({len(speech)/FS:.1f}s)")
print("Injecting: 0.5s silence + speech + 1.0s silence\n")

# 0.5s pre-silence
for _ in range(int(0.5 * FS / FRAME)):
    listener._q.put(silence.copy())
    time.sleep(FRAME / FS)

# speech frames
frames = audio_to_frames(speech)
for f in frames:
    listener._q.put(f)
    time.sleep(FRAME / FS)

# 1.0s trailing silence to trigger utterance end
for _ in range(int(1.0 * FS / FRAME)):
    listener._q.put(silence.copy())
    time.sleep(FRAME / FS)

time.sleep(1.5)
listener.stop()

# ── result ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if captured:
    print(f"PASS — VAD fired {len(captured)} utterance(s)")
    for i, u in enumerate(captured, 1):
        print(f"  {i}. {u['dur']:.2f}s  RMS={u['rms']:.4f}")
else:
    print("FAIL — VAD did not detect speech in the WAV file")
    sys.exit(1)
