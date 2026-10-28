"""Review regressions: speech timing and speaker identity must survive assembly."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import translation_dub as D


def test_nested_word_times_do_not_shorten_the_utterance():
    words = [{'start': .2, 'end': 3., 'text': ' First'},
             {'start': 1., 'end': 2., 'text': ' second'}]
    blocks, _ = D.utterances(words, [{'start': 0., 'end': 4., 'speaker': 'a'}], 4.)
    assert blocks[0]['end'] >= 3., 'The first word must retain its complete time slot'


@pytest.mark.parametrize('spans', [
    [{'start': 0., 'end': 3., 'speaker': 'a'},
     {'start': .5, 'end': 2.5, 'speaker': 'b'}],
    [{'start': 0., 'end': .75, 'speaker': 'a'},
     {'start': 2.25, 'end': 4., 'speaker': 'b'}],
])
def test_ambiguous_overlapping_speakers_are_not_arbitrarily_assigned(spans):
    words = [{'start': 1., 'end': 2., 'text': ' Hello'}]
    with pytest.raises(ValueError):
        D.utterances(words, spans, 4.)


def test_failed_diarization_preserves_translation_and_subtitles(tmp_path):
    source = tmp_path / 'speech.wav'
    sf.write(source, np.sin(np.arange(D.SR * 4) * .07) * .1, D.SR)
    synthesis = []
    ctx = SimpleNamespace(is_cancelled=lambda: False,
        text_model=lambda *a, **kw: json.dumps({'translations': ['Hello.']}),
        deps=SimpleNamespace(
            dub_words=lambda *a: [{'start': .1, 'end': 3., 'text': 'Привет.'}],
            dub_speakers=lambda *a: [],
            run_gpu_worker=lambda *a: synthesis.append(True) or (False, 'not allowed')))
    result = D.dub(ctx, str(source), 'en', str(tmp_path / 'out'))
    assert not result['path'] and result['warning']
    assert not result['audio_path'] and result['speakers'] == 0 and not synthesis
    assert 'Hello.' in result['text']
    assert 'Hello.' in Path(result['subtitle_path']).read_text('utf8')


def test_corrupt_media_is_a_readable_input_error(tmp_path):
    source = tmp_path / 'broken.mp4'
    source.write_bytes(b'not media')
    with pytest.raises(ValueError, match='[Mm]edia'):
        D.probe(source)
