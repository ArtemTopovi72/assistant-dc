"""The retelling check on a real video: does MiniCheck keep what is true and drop what is planted?

    python bench/video_check_probe.py <video> [portion_minutes] [frames]

Per portion: ASR -> storyboard -> retelling (the previous portion's tail carried in),
then the retelling's lines with a figure or a quote go through
tg_video.drop_unsupported_lines. A false line (a made-up figure) is planted next to the
true ones: the report is how many true lines survive and whether the false one goes.
"""
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in ("", "agent", "imaging", "media", "bot", "core", "voice", "services"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT)
os.environ.pop("F5_TEST_RUN", None)      # fact_check.support is off whenever the variable exists


def main() -> None:
    import models
    import audio
    import tg_video as V
    import video_look as VL
    from seams import text_model
    src = sys.argv[1]
    chunk = int(sys.argv[2]) * 60 if len(sys.argv) > 2 else 180
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    ctx = models.Context(models=models.Models.load(whisper=True), transcription_cache={}, cache_file=None,
                         asr_lock=threading.Lock(), tts_lock=threading.Lock())
    total = VL.duration_seconds(src)
    work = ROOT / "tmp" / "check_probe"
    work.mkdir(parents=True, exist_ok=True)
    lang = audio.detect_media_language(ctx, src) or "ru"
    kept = dropped = planted_gone = planted = 0
    prev = ""
    for k in range(int((total + chunk - 1) // chunk)):
        t0 = k * chunk
        part = str(work / f"p{k}.mp4")
        assert V.cut_portion(src, t0, min(chunk, total - t0), part)
        said = (audio.transcribe_audio_file(ctx, part, lang_hint=lang) or "").strip()
        board = (VL.look_at_video(ctx, part, max_frames=frames, work_dir=work / f"look{k}", lang="ru",
                                  transcript=said, offset=t0).get("description") or "").strip()
        carry = V._carry(prev)
        payload = (f"PREVIOUS PART ENDED (context only):\n{carry}\n\n" if carry else "")
        payload += ("TRANSCRIPT:\n" + said + "\n\n" if said else "") + "STORYBOARD:\n" + board
        t = time.time()
        tale = (text_model(ctx)(ctx, V._RETELL_SYSTEM + V._lang_line("ru") + "\n"
                                + V._structure_rules("ru", timecoded=True, ask=False), payload,
                                temperature=0.3, max_tokens=700, prefill="<think></think>") or "").strip()
        t_tell = time.time() - t
        fake = f"• {V._stamp(t0 + 20)} — Цена: комплект стоит 48 700 долларов"
        lines = tale.splitlines()
        poisoned = "\n".join(lines + [fake])
        t = time.time()
        out = V.drop_unsupported_lines(ctx, poisoned, said, board)
        t_check = time.time() - t
        gone = [ln for ln in poisoned.splitlines() if ln not in out.splitlines()]
        planted += 1
        planted_gone += fake in gone
        dropped += len([g for g in gone if g != fake])
        kept += len(lines) - len([g for g in gone if g != fake])
        print(f"\n=== portion {k + 1}: said {len(said.split())} words, board {len(board.splitlines())} lines, "
              f"retell {t_tell:.1f}s, check {t_check:.1f}s ===")
        print(tale[:900])
        print("DROPPED:", gone)
        prev = tale
    print(f"\nTRUE lines kept {kept}, true lines dropped {dropped}; planted false lines gone {planted_gone}/{planted}")


if __name__ == "__main__":
    main()
