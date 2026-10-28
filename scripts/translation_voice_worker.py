"""Qwen3-TTS in its existing isolated environment; one bounded job per process."""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def split_text(text):
    closing = r'''['"’”»」』】》〉）〕〗〙〛)\]}]*'''
    boundary = r'[。！？][.!?。！？]*' + closing + r'\s*|[.!?]+' + closing + r'(?:\s+|$)'
    pieces, start = [], 0
    for match in re.finditer(boundary,text):
        piece = text[start:match.end()].strip()
        if piece:
            pieces.append(piece)
        start = match.end()
    if text[start:].strip():
        pieces.append(text[start:].strip())
    return pieces


def render(model, jobs, temp):
    import numpy as np
    import soundfile as sf
    prompts, work = {}, []
    for i, job in enumerate(jobs):
        source = str(Path(job['source']).resolve())
        if source not in prompts:
            ref = str(Path(temp) / f'reference_{len(prompts)}.wav')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', source, '-t', '12',
                            '-vn', '-ac', '1', '-ar', '24000', ref], check=True, timeout=60)
            data, sr = sf.read(ref)
            if len(data) < sr * job.get('min_reference_seconds', 3):
                raise ValueError('Too little reference speech.')
            prompts[source] = model.create_voice_clone_prompt(ref_audio=ref, x_vector_only_mode=True)
        pieces = split_text(job['text'])
        if not pieces:
            raise ValueError('No text to synthesize.')
        work.extend((i,piece,job['language'],prompts[source][0]) for piece in pieces)
    audio = [[] for _ in jobs]
    rate = None
    for at in range(0,len(work),4):
        batch = work[at:at+4]
        print(f'{at}/{len(work)} [synthesizing]', flush=True)
        wavs, sr = model.generate_voice_clone(text=[item[1] for item in batch],
                language=[item[2] for item in batch],voice_clone_prompt=[item[3] for item in batch],max_new_tokens=2048,
                repetition_penalty=1.15)
        if len(wavs)!=len(batch) or (rate is not None and sr!=rate):
            raise ValueError('Invalid voice synthesis batch.')
        rate = sr
        for item,wav in zip(batch,wavs):
            if not len(wav) or not np.isfinite(wav).all():
                raise ValueError('Empty or invalid synthesized speech.')
            audio[item[0]].append(wav)
    for job,parts in zip(jobs,audio):
        joined = [value for n,part in enumerate(parts)
                  for value in ((np.zeros(round(rate*.08),dtype=np.float32),part) if n else (part,))]
        sf.write(job['output'],np.concatenate(joined),rate)
    print(f'{len(work)}/{len(work)} [complete]', flush=True)


def main():
    job = json.loads(Path(sys.argv[1]).read_text('utf8'))
    import torch
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    jobs = job.get('segments') or [job]
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 500:
        raise ValueError('Invalid synthesis batch.')
    model = Qwen3TTSModel.from_pretrained(job['model'], device_map='cuda:0',
            dtype=torch.bfloat16, attn_implementation='sdpa')
    with tempfile.TemporaryDirectory(prefix='translation_voice_') as temp:
        render(model, jobs, temp)


if __name__ == '__main__':
    main()
