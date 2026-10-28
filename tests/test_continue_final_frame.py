import subprocess

import numpy as np
import pytest
from PIL import Image

from tg_continue import seed_frame, cut_tail


def test_seed_preserves_change_in_final_frame(tmp_path):
    clip = tmp_path / 'last_change.mp4'
    frame = tmp_path / 'seed.jpg'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=160x120:rate=24:duration=1', '-vf',
                    "drawbox=x=0:y=0:w=160:h=120:color=red:t=fill:enable='gte(t,0.95)'",
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(clip)], check=True)
    assert seed_frame(str(clip), str(frame))
    pixel = np.asarray(Image.open(frame).convert('RGB'))[60, 80]
    assert pixel[0] > 180 and pixel[1] < 80 and pixel[2] < 80
    tail = tmp_path / 'tail.mp4'
    assert cut_tail(str(clip), str(tail))
    tail_frame = tmp_path / 'tail_end.jpg'
    assert seed_frame(str(tail), str(tail_frame))
    pixel = np.asarray(Image.open(tail_frame).convert('RGB'))[60, 80]
    assert pixel[0] > 180 and pixel[1] < 80 and pixel[2] < 80


@pytest.mark.parametrize('extension', ['mp4', 'mkv'])
def test_audio_longer_than_video_does_not_hide_the_final_frame(tmp_path, extension):
    from video import probe
    clip = tmp_path / f'long_audio.{extension}'
    frame, tail = tmp_path / 'seed.jpg', tmp_path / 'tail.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=red:size=160x120:rate=24:duration=1', '-f', 'lavfi', '-i',
                    'sine=frequency=400:duration=6', '-c:v', 'libx264', '-c:a', 'aac', str(clip)], check=True)
    assert seed_frame(str(clip), str(frame))
    assert cut_tail(str(clip), str(tail))
    info = probe(str(tail))
    assert info['width'] == 160 and info['has_audio'] and 1.95 <= info['seconds'] <= 2.1
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', '1.4', '-t', '0.4', '-i', str(tail),
                                   '-vn', '-ac', '1', '-ar', '8000', '-f', 'f32le', '-'])
    assert np.max(np.abs(np.frombuffer(raw, dtype=np.float32))) < 0.01


def test_short_silent_clip_has_two_second_reference(tmp_path):
    from video import probe
    clip, tail = tmp_path / 'short.mp4', tmp_path / 'tail.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=red:size=160x120:rate=24:duration=0.5', '-c:v', 'libx264', str(clip)], check=True)
    assert cut_tail(str(clip), str(tail))
    info = probe(str(tail))
    assert not info['has_audio'] and 1.95 <= info['seconds'] <= 2.1


def test_failed_frame_extraction_does_not_replace_existing_output(tmp_path):
    frame = tmp_path / 'existing.jpg'
    frame.write_bytes(b'old')
    assert not seed_frame(str(tmp_path / 'missing.mp4'), str(frame))
    assert frame.read_bytes() == b'old'
