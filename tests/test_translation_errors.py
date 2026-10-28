"""Error edges of the translator family: what a user sees when things break.

A raw engine string (a disk path, an English traceback line) must never reach
a user; a missing file must read as a readable input error; a heading that
cannot fit one message is rejected outright instead of being cut mid-tag.

Run: venv/Scripts/python.exe -m pytest tests/test_translation_errors.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'media'))
sys.path.insert(0, str(ROOT / 'bot'))

import translator as TR
import translation_dub as D
from tg_translation_format import format_sections


def test_unexpected_engine_failure_hides_the_raw_exception(tmp_path):
    source = tmp_path / 'speech.wav'
    sf.write(source, np.sin(np.arange(D.SR * 4) * .07) * .1, D.SR)

    def boom(*a, **kw):
        raise RuntimeError('engine exploded at C:/models/secret/model.safetensors')

    ctx = SimpleNamespace(is_cancelled=lambda: False,
        text_model=lambda *a, **kw: json.dumps({'translations': ['Hello.']}),
        deps=SimpleNamespace(
            dub_words=lambda *a: [{'start': .1, 'end': 3., 'text': 'Привет.'}],
            dub_speakers=lambda *a: [{'start': 0., 'end': 3.5, 'speaker': 'a'}],
            run_gpu_worker=boom))
    result = D.dub(ctx, str(source), 'en', str(tmp_path / 'out'))
    assert result['warning'] == 'Could not process this media file.'
    assert 'C:/' not in result['warning'] and 'exploded' not in result['warning']
    assert result['subtitle_path'] and Path(result['subtitle_path']).is_file(), \
        'the subtitles survive an engine failure'


def test_curated_failures_keep_their_message_and_translation():
    message = 'Voice synthesis failed; subtitles are preserved.'
    assert TR.curated_error(ValueError(message)) == message
    assert TR.curated_error(RuntimeError('C:/raw/path')) == ''
    assert 'Синтез речи' in TR.error_text(ValueError(message), 'ru'), \
        'every curated message must have a Russian rendering'


def test_missing_media_file_is_a_readable_input_error(tmp_path):
    ctx = SimpleNamespace(is_cancelled=lambda: False)
    with pytest.raises(ValueError) as missing_dub:
        D.dub(ctx, str(tmp_path / 'nope.mp4'), 'en', str(tmp_path / 'out'))
    assert str(missing_dub.value) == 'Could not process this media file.'
    assert TR.curated_error(missing_dub.value), 'and it is a curated message'
    with pytest.raises(ValueError) as missing_voice:
        TR.transcribe(ctx, str(tmp_path / 'nope.ogg'))
    assert str(missing_voice.value) == 'Could not process this media file.'


def test_a_heading_that_cannot_fit_one_message_is_rejected():
    with pytest.raises(ValueError):
        format_sections([('x' * 4000, 'body')])


def test_multi_chunk_output_stays_under_the_limit_and_keeps_headers():
    body = 'аВ!' * 1300                      # escapes stay cheap, size is the point
    chunks = format_sections([('Заголовок', body), ('Второй', 'текст')])
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.encode('utf-16-le')) // 2 <= 3500
        assert chunk.startswith('<b>'), 'every chunk opens with a valid header'
    assert ''.join(c for c in chunks if 'Заголовок' in c).count('аВ!') >= 1300
