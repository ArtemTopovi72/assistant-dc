import numpy as np
import torch
import audio


def test_every_synthesized_batch_has_one_safe_ending(monkeypatch):
    import torchaudio
    from f5_tts.infer import utils_infer as f5
    seen = []
    monkeypatch.setattr(torchaudio, 'load', lambda _: (torch.zeros((1, 24000 * 10)), 24000))
    monkeypatch.setattr(audio, 'TTS_END_PADDING', '.')
    def render(ref, text, batches, model, vocoder, **kwargs):
        seen.extend(batches)
        yield np.zeros(10), 24000, None
    monkeypatch.setattr(f5, 'infer_batch_process', render)
    text = ('Первая длинная фраза про погоду. Вторая длинная фраза про поездку. ' * 5).strip()
    audio._infer_f5_padded('ref.wav', 'образец голоса достаточно длинный', text, object(), object(), speed=1)
    assert len(seen) > 1
    assert all(part.endswith('.' * 15) and not part.endswith('.' * 16) for part in seen)
    assert 'Первая' in seen[0] and 'поездку' in seen[-1]


def test_teacher_russian_reaches_the_real_f5_batch_wrapper(tmp_path, monkeypatch):
    import threading
    import wave
    from types import SimpleNamespace as NS
    import delivery
    import translator
    import torchaudio
    from f5_tts.infer import utils_infer as f5

    ref = tmp_path / 'ref.wav'
    def wav(path):
        with wave.open(str(path), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b'\0\0' * 12000)
    wav(ref)
    model_dir = tmp_path / 'models_ext' / 'qwen3_tts_base'; model_dir.mkdir(parents=True)
    (model_dir / 'model.safetensors').touch()
    python = tmp_path / 'python.exe'; python.touch()
    seen = []
    monkeypatch.setattr(audio, 'stress_plus', lambda ctx, text: text)
    monkeypatch.setattr(delivery, 'enabled', lambda: False)
    monkeypatch.setattr(torchaudio, 'load', lambda _: (torch.zeros((1, 24000 * 10)), 24000))
    def render(ref_audio, ref_text, batches, model, vocoder, **kwargs):
        seen.extend(batches)
        yield np.zeros(24000, dtype=np.float32), 24000, None
    monkeypatch.setattr(f5, 'infer_batch_process', render)
    def qwen(ctx, python, script, job, label, timeout, **kwargs):
        for item in job['segments']: wav(item['output'])
        return True, ''
    deps = NS(preprocess_ref_audio_text=lambda path, text: (path, text),
              get_actor_ref_and_speed=lambda *a: (str(ref), 'образец голоса достаточно длинный', 1.0),
              resolve_ref_audio=lambda path: path,
              transcribe_audio_file=lambda *a, **kw: 'образец голоса достаточно длинный', run_gpu_worker=qwen)
    ctx = NS(models=NS(tts_model=object(), vocoder=object(), accentor_loaded=False),
             tts_lock=threading.Lock(), deps=deps,
             modules={'config': NS(BASE_DIR=tmp_path, venv_python=lambda _: str(python))},
             is_cancelled=lambda: False, model_name='')
    result = translator.lesson_speech(ctx, 'Hallo!', 'de', 'Первое предложение. Второе предложение. Третье предложение.',
                                      'ru', str(tmp_path / 'out'), reference=str(ref))
    assert len(result['paths']) == 2 and len(seen) > 1
    assert all(part.endswith('.' * 15) and not part.endswith('.' * 16) for part in seen)
