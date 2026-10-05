"""What was actually sung:  venv/Scripts/python.exe bench/whisper_check.py audio [lang]"""
import sys
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
segs, info = m.transcribe(sys.argv[1], language=sys.argv[2] if len(sys.argv) > 2 else "ru", word_timestamps=False, vad_filter=False, condition_on_previous_text=False)
for s in segs:
    print(f"[{s.start:6.1f}-{s.end:6.1f}] {s.text.strip()}")
