"""Qwen3-TTS in its existing isolated environment; one bounded job per process."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def render(model, jobs, temp):
    import soundfile as sf
    prompts = {}
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
        print(f'{i}/{len(jobs)} [synthesizing]', flush=True)
        wavs, sr = model.generate_voice_clone(text=job['text'], language=job['language'],
                voice_clone_prompt=prompts[source], max_new_tokens=2048,
                repetition_penalty=1.15)
        sf.write(job['output'], wavs[0], sr)
    print(f'{len(jobs)}/{len(jobs)} [complete]', flush=True)


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
