from pathlib import Path
from types import SimpleNamespace
import importlib.util

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location('translation_voice_worker',
    Path(__file__).resolve().parents[1] / 'scripts' / 'translation_voice_worker.py')
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def test_batch_reuses_only_the_same_speakers_reference(tmp_path):
    refs, generated = [], []
    def prompt(ref_audio, **kw):
        refs.append(Path(ref_audio).read_bytes())
        return [len(refs)]
    def generate(**kw):
        generated.append(kw)
        return [np.zeros(2400, dtype=np.float32)], 24000
    model = SimpleNamespace(create_voice_clone_prompt=prompt, generate_voice_clone=generate)
    paths = []
    for i in range(2):
        p = tmp_path / f'ref{i}.wav'
        sf.write(p, np.sin(np.arange(24000*4) * (.02+i*.01)).astype(np.float32), 24000)
        paths.append(str(p))
    jobs = [{'source': paths[s], 'text': f'phrase {i}', 'language': lang,
             'output': str(tmp_path/f'out{i}.wav')}
            for i, (s, lang) in enumerate(((0,'English'),(1,'French'),(0,'Japanese')))]
    W.render(model, jobs, str(tmp_path))
    assert len(refs) == 2
    assert [x['voice_clone_prompt'] for x in generated] == [[1],[2],[1]]
    assert [x['language'] for x in generated] == ['English','French','Japanese']
    assert all(Path(j['output']).exists() for j in jobs)
