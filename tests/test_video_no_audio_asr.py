import os
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)] + [str(ROOT / d) for d in ('core', 'agent', 'bot', 'voice', 'media', 'imaging', 'research', 'knowledge', 'services')]
import tg_tasks


def test_video_without_audio_does_not_call_asr(tmp_path):
    clip = tmp_path / 'silent.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=black:s=64x64:d=1', '-c:v', 'libx264', str(clip)], check=True)
    calls = []
    ctx = types.SimpleNamespace(deps=types.SimpleNamespace(
        detect_media_language=lambda *a, **k: calls.append('language') or 'ru',
        transcribe_audio_file=lambda *a, **k: calls.append('asr') or ''))
    runner = tg_tasks.TaskRunnerMixin()
    assert runner._transcribe(ctx, 'unused', 'video', data=clip.read_bytes()) == ''
    assert calls == []


def test_other_decode_errors_keep_asr_fallback(monkeypatch):
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'ffmpeg', stderr=b'unsupported codec')
    monkeypatch.setattr(tg_tasks.subprocess, 'run', failed)
    calls = []
    ctx = types.SimpleNamespace(deps=types.SimpleNamespace(
        detect_media_language=lambda *a, **k: calls.append('language') or 'ru',
        transcribe_audio_file=lambda *a, **k: calls.append('asr') or 'one two three'))
    runner = tg_tasks.TaskRunnerMixin()
    assert runner._transcribe(ctx, 'unused', 'video', data=b'input') == 'one two three'
    assert calls == ['language', 'asr']
