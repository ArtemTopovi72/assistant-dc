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
        return [np.zeros(2400, dtype=np.float32) for _ in kw['text']], 24000
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
    assert len(generated)==1
    assert generated[0]['voice_clone_prompt'] == [1,2,1]
    assert generated[0]['language'] == ['English','French','Japanese']
    assert all(Path(j['output']).exists() for j in jobs)


def test_sentence_chunks_preserve_words_numbers_and_cjk_punctuation():
    text='At 3.14 we meet. Потом идём домой! 今天好。明天见！'
    parts=W.split_text(text)
    assert len(parts)==4 and parts[0]=='At 3.14 we meet.'
    assert ''.join(''.join(parts).split())==''.join(text.split())


def test_batch_is_bounded_and_keeps_each_outputs_own_speaker_and_order(tmp_path):
    source=tmp_path/'ref.wav';sf.write(source,np.zeros(24000*4),24000)
    batches=[]
    def generate(**kw):
        batches.append(kw)
        return [np.full(240, float(text[0])/10,dtype=np.float32) for text in kw['text']],24000
    model=SimpleNamespace(create_voice_clone_prompt=lambda **kw:['same'],generate_voice_clone=generate)
    job={'source':str(source),'text':'1. 2. 3. 4. 5. 6.','language':'English','output':str(tmp_path/'out.wav')}
    W.render(model,[job],str(tmp_path))
    assert [len(batch['text']) for batch in batches]==[4,2]
    audio,sr=sf.read(job['output'])
    stride=240+int(sr*.08)
    assert all(abs(audio[i*stride]-n/10)<.001 for i,n in enumerate(range(1,7)))


def test_incomplete_batch_is_not_silently_accepted(tmp_path):
    source=tmp_path/'ref.wav';sf.write(source,np.zeros(24000*4),24000)
    model=SimpleNamespace(create_voice_clone_prompt=lambda **kw:['same'],generate_voice_clone=lambda **kw:([],24000))
    import pytest
    with pytest.raises(ValueError):
        W.render(model,[{'source':str(source),'text':'Hello.','language':'English','output':str(tmp_path/'out.wav')}],str(tmp_path))
    assert not (tmp_path/'out.wav').exists()
