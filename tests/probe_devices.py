import sounddevice as sd
import numpy as np

print("Testing which input devices open at 16kHz...")
for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0:
        try:
            s = sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=512, device=i)
            s.start()
            data, _ = s.read(512)
            rms = float(np.sqrt(np.mean(data**2)))
            s.stop()
            s.close()
            print(f"  [{i}] OK   RMS={rms:.5f}  {dev['name']}")
        except Exception as e:
            print(f"  [{i}] FAIL {dev['name']}  ({e})")
