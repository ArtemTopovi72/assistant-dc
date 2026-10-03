"""
Standalone VAD test — speak into the mic and watch utterances get captured.
Usage: .\\venv\\Scripts\\python.exe tests\\test_vad.py [seconds]
Default: listens for 15 seconds then exits.
"""
import sys
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import time
import threading
import numpy as np
import sounddevice as sd

sys.path.insert(0, ".")

LISTEN_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
DEVICE = int(sys.argv[2]) if len(sys.argv) > 2 else None  # None = system default

# ── show devices ────────────────────────────────────────────────────────────
print("Available input devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0:
        marker = " <-- default" if i == sd.default.device[0] else ""
        print(f"  [{i}] {dev['name']}{marker}")
print()

# ── shared state ─────────────────────────────────────────────────────────────
captured = []
_last_rms = [0.0]
_rms_lock = threading.Lock()

def on_utterance(audio: np.ndarray):
    duration = len(audio) / 16000
    rms = float(np.sqrt(np.mean(audio ** 2)))
    n = len(captured) + 1
    captured.append({"duration": duration, "rms": rms})
    print(f"\n  >>> utterance {n}: {duration:.2f}s  peak-RMS={rms:.4f}", flush=True)

# ── RMS monitor stream (reads mic independently to show levels) ──────────────
def _rms_callback(indata, frames, time_info, status):
    rms = float(np.sqrt(np.mean(indata ** 2)))
    with _rms_lock:
        _last_rms[0] = rms

if DEVICE is not None:
    print(f"Using device [{DEVICE}]: {sd.query_devices(DEVICE)['name']}\n")
else:
    print(f"Using default input device\n")

monitor = sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                          blocksize=512, callback=_rms_callback, device=DEVICE)
monitor.start()

# ── VAD listener ─────────────────────────────────────────────────────────────
print("Loading Silero VAD model...")
from audio import VadListener

listener = VadListener(on_utterance=on_utterance, device=DEVICE)
listener.start()
print(f"Listening for {LISTEN_SECONDS}s — speak into your mic now.\n")

t0 = time.time()
try:
    while time.time() - t0 < LISTEN_SECONDS:
        with _rms_lock:
            rms = _last_rms[0]
        bar = int(min(rms * 3000, 40))
        remaining = int(LISTEN_SECONDS - (time.time() - t0))
        print(f"\r  [{remaining:2d}s] mic |{'#'*bar}{' '*(40-bar)}| RMS={rms:.5f}  utterances={len(captured)}", end="", flush=True)
        time.sleep(0.15)
except KeyboardInterrupt:
    pass
finally:
    listener.stop()
    monitor.stop()
    monitor.close()

print(f"\n\nDone. Captured {len(captured)} utterance(s).")
for i, u in enumerate(captured, 1):
    print(f"  {i}. {u['duration']:.2f}s  RMS={u['rms']:.4f}")

if not captured:
    print("\nNo speech detected.")
    print("  - If mic RMS stayed near 0: mic not captured (wrong device?)")
    print("  - If RMS was non-zero but no utterance: lower VAD_THRESHOLD (currently 0.5)")
    print("    e.g.  VAD_THRESHOLD=0.3 .\\venv\\Scripts\\python.exe tests\\test_vad.py")
