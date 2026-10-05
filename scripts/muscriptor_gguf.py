"""MuScriptor through the open GGUF build (the official weights are gated): a llama.cpp server holds the 1.4B
transformer, the conditioning (mel + instrument group) is computed here. Same surface as muscriptor.TranscriptionModel:
`transcribe(wav, instruments=None)` yields NoteStartEvent / NoteEndEvent (the GGUF repo's own event classes).

    with Server() as srv:  model = GgufModel();  for ev in model.transcribe(wav, ["voice"]): ...
"""
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_DIR = os.path.join(ROOT, "models_ext", "muscriptor_gguf")
SERVER_EXE = os.path.join(ROOT, "models_ext", "muscriptor_llama", "build", "bin", "llama-server.exe")
MODEL = os.path.join(GGUF_DIR, "model_fp16_8k.gguf")
PORT = int(os.getenv("MUSCRIPTOR_PORT", "18081"))
sys.path.insert(0, GGUF_DIR)


class Server:
    """llama-server for the model; reuses one already listening on the port."""

    def __init__(self, ngl=None):
        self.ngl = os.getenv("MUSCRIPTOR_NGL", "99") if ngl is None else str(ngl)
        self.proc = None

    def _up(self):
        try:
            return requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def __enter__(self):
        if not self._up():
            self.proc = subprocess.Popen(
                [SERVER_EXE, "-m", MODEL, "--port", str(PORT), "-ngl", self.ngl, "--parallel", "4"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for _ in range(120):
                if self._up():
                    break
                time.sleep(1)
            else:
                self.__exit__()
                raise RuntimeError("llama-server did not come up")
        return self

    def __exit__(self, *a):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
            self.proc = None


class GgufModel:
    def __init__(self, device="cpu"):
        import muscriptor_cli as m
        self._m = m

        def load(path, target_sr=m._SAMPLE_RATE):               # torchaudio here has no torchcodec
            import soundfile as sf
            import torch
            import torchaudio
            x, sr = sf.read(path, dtype="float32", always_2d=True)
            w = torch.from_numpy(x.mean(1))[None]
            return torchaudio.functional.resample(w, sr, target_sr) if sr != target_sr else w
        m._load_audio = load
        self.llama = m.MuScriptorLlama(os.path.join(GGUF_DIR, "conditioning_weights.safetensors"),
                                       server_url=f"http://127.0.0.1:{PORT}", device=device, parallel=4)

    def transcribe(self, wav, instruments=None):
        from muscriptor.tokenizer.mt3 import instrument_group_from_names
        group = instrument_group_from_names(instruments) if instruments else None
        raw = self.llama.transcribe(wav, instrument_group=group)
        for ev in self._m._decode_model_tokens(raw, self.llama.vocab, self._m._instrument_for_program):
            if isinstance(ev, (self._m.NoteStartEvent, self._m.NoteEndEvent)):
                yield ev


if __name__ == "__main__":
    with Server():
        mdl = GgufModel()
        t = time.time()
        notes = [e for e in mdl.transcribe(sys.argv[1], sys.argv[2:] or None) if isinstance(e, mdl._m.NoteEndEvent)]
        print(len(notes), "notes in", round(time.time() - t), "s")
        for e in notes[:8]:
            s = e.start_event
            print(round(s.start_time, 2), round(e.end_time, 2), s.pitch, s.instrument)
