"""Speaker diarization worker. Runs in venv_diar (transformers master + CPU torch).

    venv_diar/Scripts/python scripts/diar_worker.py <wav16k> [model_dir]
Prints one JSON line: [{"speaker": 0, "start": 0.0, "end": 1.2}, ...]
"""
import json, sys

import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

path = sys.argv[1]
model_id = sys.argv[2] if len(sys.argv) > 2 else "nvidia/Nemotron-3-Diarization"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id).eval()
sr = processor.feature_extractor.sampling_rate
audio = load_audio(path, sampling_rate=sr)
inputs = processor(audio, sampling_rate=sr)
with torch.inference_mode():
    logits = model(**inputs).logits
segs = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
print(json.dumps([{"speaker": int(s["Speaker"]), "start": float(s["Start"]), "end": float(s["End"])}
                  for s in segs]))
