"""Qwen3-TTS in its existing isolated environment; one bounded job per process."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    job = json.loads(Path(sys.argv[1]).read_text('utf8'))
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='translation_voice_') as temp:
        ref = str(Path(temp) / 'reference.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', job['source'], '-t', '12',
                        '-vn', '-ac', '1', '-ar', '24000', ref], check=True, timeout=60)
        data, sr = sf.read(ref)
        if len(data) < sr * 3:
            raise ValueError('At least three seconds of speech are needed.')
        model = Qwen3TTSModel.from_pretrained(job['model'], device_map='cuda:0',
                dtype=torch.bfloat16, attn_implementation='sdpa')
        wavs, sr = model.generate_voice_clone(text=job['text'], language=job['language'],
                ref_audio=ref, x_vector_only_mode=True, max_new_tokens=2048,
                repetition_penalty=1.15)
        sf.write(job['output'], wavs[0], sr)


if __name__ == '__main__':
    main()
