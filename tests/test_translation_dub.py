from types import SimpleNamespace
from pathlib import Path
import json
import subprocess

import numpy as np
import pytest
import soundfile as sf
import translation_dub as D


def test_words_switch_voice_on_speaker_boundaries_without_losing_text():
    words = [{'start': .2, 'end': 1., 'text': ' Hello'}, {'start': 1., 'end': 1.5, 'text': ' there.'},
             {'start': 3., 'end': 4., 'text': ' Good morning.'}]
    speakers = [{'start':0.,'end':2.,'speaker':0},{'start':2.8,'end':4.1,'speaker':1}]
    result, _ = D.utterances(words, speakers, 5.)
    assert [x['speaker'] for x in result] == ['0','1']
    assert [x['text'] for x in result] == ['Hello there.','Good morning.']


def test_invalid_or_unassigned_speech_is_not_silently_dropped():
    with pytest.raises(ValueError):
        D.utterances([{'start':float('nan'),'end':2.,'text':'hello'}], [], 4.)
    with pytest.raises(ValueError):
        D.utterances([{'start':0.,'end':2.,'text':'hello'}], [], 4.)


def fixture(tmp_path, video=False):
    source = tmp_path/'source.wav'
    sf.write(source, np.sin(np.arange(16000*8)*.04).astype(np.float32)*.1, 16000)
    if video:
        clip=tmp_path/'source.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=160x120:r=10:d=8',
            '-i',str(source),'-c:v','libx264','-c:a','aac','-shortest',str(clip)],check=True)
        source=clip
    ctx=SimpleNamespace(is_cancelled=lambda:False)
    def worker(ctx, python, script, job, label, timeout):
        assert len(job['segments'])==2
        assert len({x['source'] for x in job['segments']})==2
        for item in job['segments']:
            sf.write(item['output'], np.sin(np.arange(24000*4)*.07).astype(np.float32)*.2,24000)
        return True,''
    ctx.text_model=lambda *a,**kw: json.dumps({'translations':['Hello one.','Hello two.']})
    ctx.deps=SimpleNamespace(dub_words=lambda *a: [
        {'start':.2,'end':3.2,'text':'Первый говорит.'},{'start':4.2,'end':7.2,'text':'Второй отвечает.'}],
        dub_speakers=lambda *a: [
        {'start':.1,'end':3.3,'speaker':0},{'start':4.1,'end':7.3,'speaker':1}],run_gpu_worker=worker)
    return ctx,source


@pytest.mark.parametrize('video',[False,True])
def test_dub_preserves_timeline_source_and_both_voices(tmp_path,video):
    ctx,source=fixture(tmp_path,video)
    before=source.read_bytes()
    result=D.dub(ctx,str(source),'en',str(tmp_path/'out'))
    assert source.read_bytes()==before
    assert abs(D.probe(result['path'])['duration']-8.)<.15
    assert abs(sf.info(result['audio_path']).duration-8.)<.01
    assert result['speakers']==2 and Path(result['subtitle_path']).exists()
    subtitle=Path(result['subtitle_path']).read_text('utf8')
    assert '00:00:00,200 --> 00:00:03,200' in subtitle
    assert 'Hello one.' in subtitle and 'Hello two.' in subtitle
    if video:
        def frames(path):
            raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-f','framemd5','-'])
            return [line.split(b',')[-1].strip() for line in raw.splitlines() if line and not line.startswith(b'#')]
        assert frames(source)==frames(result['path'])


def test_failed_synthesis_keeps_subtitles_without_claiming_a_dub(tmp_path):
    ctx,source=fixture(tmp_path)
    ctx.deps.run_gpu_worker=lambda *a: (False,'failed')
    result=D.dub(ctx,str(source),'en',str(tmp_path/'out'))
    assert not result['path'] and result['warning']
    assert Path(result['subtitle_path']).exists() and 'Hello one.' in result['text']


def test_cancel_before_work_starts_never_opens_source(tmp_path):
    with pytest.raises(InterruptedError):
        D.dub(SimpleNamespace(is_cancelled=lambda:True),'missing.mp4','en',str(tmp_path))


def test_cancel_reaps_an_external_media_process():
    import threading, time, sys
    event=threading.Event(); timer=threading.Timer(.3,event.set); timer.start()
    start=time.monotonic()
    try:
        with pytest.raises(InterruptedError):
            D._run(SimpleNamespace(is_cancelled=event.is_set),
                   [sys.executable,'-c','import time; time.sleep(30)'])
        assert time.monotonic()-start < 3
    finally:
        timer.cancel()


def test_reference_never_contains_an_overlapping_other_speaker(tmp_path):
    audio=np.full(D.SR*3,.1,dtype=np.float32)
    audio[D.SR:D.SR*2]=.9
    path=tmp_path/'ref.wav'
    D._reference(audio,[{'start':0.,'end':3.,'speaker':'a'},
                       {'start':1.,'end':2.,'speaker':'b'}],'a',str(path))
    ref,sr=sf.read(path)
    assert len(ref)==D.SR*2 and np.max(np.abs(ref))<.2
