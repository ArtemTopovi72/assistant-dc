"""Real clip through the long-video path: ASR time, look time at two frame budgets.

    python bench/video_real_probe.py <video> [frames ...]     # default 12 40

No ground truth here -- it reports speed and prints what each budget kept so a
human can judge; the planted-fact scoring lives in video_retell_eval.py.
"""
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in ("", "agent", "imaging", "media", "bot", "core", "voice"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT)
os.environ.setdefault("F5_TEST_RUN", "0")


def main() -> None:
    import models
    import audio
    import video_look as VL
    src = sys.argv[1]
    budgets = [int(x) for x in sys.argv[2:]] or [12, 40]
    ctx = models.Context(models=models.Models.load(whisper=True), transcription_cache={}, cache_file=None,
                         asr_lock=threading.Lock(), tts_lock=threading.Lock())
    total = VL.duration_seconds(src)
    print(f"{src}: {total:.0f}s")
    t0 = time.time()
    lang = audio.detect_media_language(ctx, src) or "ru"
    t_lang = time.time() - t0
    t0 = time.time()
    said = (audio.transcribe_audio_file(ctx, src, lang_hint=lang) or "").strip()
    print(f"ASR {time.time() - t0:.1f}s (lang detect {t_lang:.1f}s, {lang}): {len(said.split())} words")
    print("SAID:", said[:600].replace("\n", " "))
    for n in budgets:
        t0 = time.time()
        seen = VL.look_at_video(ctx, src, max_frames=n, work_dir=ROOT / "tmp" / f"real_probe_{n}",
                                lang="ru", transcript=said, offset=0)
        dt = time.time() - t0
        desc = (seen.get("description") or "").strip()
        print(f"\n=== frames={n}: kept {len(seen.get('times') or [])} frames, {dt:.1f}s "
              f"({dt / (total / 60):.1f}s per video minute), {len(desc.split())} words ===")
        print(desc[:1800])


if __name__ == "__main__":
    main()
