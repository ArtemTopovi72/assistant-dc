"""Render one MuLaCover cover. Runs in venv_mula (its numpy/torchao pins clash with the app).

    venv_mula/Scripts/python scripts/mulacover_render.py job.json

job.json: {"ref": wav, "lyrics": str, "tags": str, "seed": int, "out": path, "vocals": wav (optional)}.
The model is CC BY-NC: personal, non-commercial use only.

The melody the cover follows is read by MuScriptor (Kyutai/Mirelo, a far better transcriber than the YourMT3 that ships
with MuLaCover) from the separated VOCAL stem when the job has one, drums from the full mix; chords stay with ChordNet.
Any failure of that path falls back to MuLaCover's own transcription.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.getenv("MULACOVER_CKPT", os.path.join(ROOT, "models_ext", "mulacover_ckpt"))
MUSCRIPTOR = os.getenv("MUSCRIPTOR_SIZE", "large")


def _notes(model, wav, instruments=None):
    """MuScriptor events -> [(onset, offset, pitch, instrument)] in seconds."""
    from muscriptor import NoteStartEvent, NoteEndEvent
    out = []
    for ev in model.transcribe(wav, instruments=instruments):
        if isinstance(ev, NoteEndEvent):
            s = ev.start_event
            out.append((s.start_time, max(ev.end_time, s.start_time), s.pitch, s.instrument))
    return out


def install_muscriptor(job):
    """Swap MuLaCover's YourMT3 melody step for MuScriptor (vocal stem -> melody, full mix -> drums)."""
    import torch
    from muscriptor import TranscriptionModel
    import mulacover._symbolic_transcription.melody as melody_mod
    original = melody_mod.MelodyTranscriber
    vocals = job.get("vocals")

    class MuScriptorMelody:
        def __init__(self, checkpoint_path, device, dtype):
            self._fallback = lambda: original(checkpoint_path, device, dtype)

        def transcribe(self, audio_path):
            try:
                model = TranscriptionModel.load_model(MUSCRIPTOR, device="cuda:0" if torch.cuda.is_available() else "cpu")
                mix = _notes(model, str(audio_path))
                if vocals and os.path.isfile(vocals):
                    voice = _notes(model, vocals, instruments=["voice"])
                else:
                    voice = [n for n in mix if n[3] == "voice"]
                del model
                torch.cuda.empty_cache()
                if len(voice) < 8:
                    raise RuntimeError(f"MuScriptor heard {len(voice)} sung notes")
                print(f"muscriptor: {len(voice)} sung notes, {sum(n[3] == 'drums' for n in mix)} drum hits", flush=True)
                rows = [dict(onset=o, offset=f, pitch=p, is_drum=False, program=100) for o, f, p, _ in voice]
                rows += [dict(onset=o, offset=f, pitch=p, is_drum=True, program=128) for o, f, p, i in mix if i == "drums"]
                return rows
            except Exception as exc:                             # noqa: BLE001 -- the shipped transcriber still works
                print("muscriptor failed, using YourMT3:", repr(exc), flush=True)
                return self._fallback().transcribe(audio_path)

    melody_mod.MelodyTranscriber = MuScriptorMelody


def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    import torch
    from mulacover import MuLaCoverGenPipeline
    if os.getenv("MULACOVER_YOURMT3") != "1":
        try:
            install_muscriptor(job)
        except Exception as exc:                                 # noqa: BLE001
            print("muscriptor not installed:", repr(exc), flush=True)
    pipe = MuLaCoverGenPipeline.from_pretrained(
        CKPT, device=torch.device("cuda:0"),
        dtype={"mulacover": torch.bfloat16, "codec": torch.float32,
               "qwen": torch.float32, "transcriptor": torch.float32},
        lazy_load=True)
    with tempfile.TemporaryDirectory() as td:
        lp, tp = os.path.join(td, "lyrics.txt"), os.path.join(td, "tags.txt")
        open(lp, "w", encoding="utf-8").write(job["lyrics"])
        open(tp, "w", encoding="utf-8").write(job["tags"])
        torch.manual_seed(int(job["seed"]))
        pipe({"ref_audio": job["ref"], "bpm": job.get("bpm"), "lyrics": lp, "tags": tp},
             save_path=job["out"], temperature=1.0, topk=250, cfg_scale=1.5)
    print("OK", job["out"], flush=True)


if __name__ == "__main__":
    main()
