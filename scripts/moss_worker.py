"""MOSS-Transcribe-Diarize 0.9B (one pass: who / when / what). Runs in venv_diar (transformers >= 5.6, CPU torch).

    venv_diar/Scripts/python scripts/moss_worker.py <wav> [wav ...]
One JSON line per file: {"wav": ..., "segments": [{"speaker": "S01", "start": s, "end": e, "text": ...}], "sec": t}
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


def main() -> None:
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype="auto").to(torch.float32).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    for wav in sys.argv[1:]:
        t = time.time()
        res = generate_transcription(model, processor, build_transcription_messages(wav), max_new_tokens=2048,
                                     do_sample=False, device=device, dtype=torch.float32)
        segs = [{"speaker": str(s.speaker), "start": float(s.start), "end": float(s.end), "text": s.text}
                for s in parse_transcript(res["text"])]
        print(json.dumps({"wav": wav, "segments": segs, "sec": round(time.time() - t, 1)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
