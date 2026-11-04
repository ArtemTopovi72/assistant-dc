import shutil
import subprocess

import pytest
import video


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg required')
@pytest.mark.parametrize(('duration', 'expected'), [(3, 3), (30, 25)])
def test_voice_sample_keeps_up_to_last_25_seconds(tmp_path, duration, expected):
    source = tmp_path / 'source.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'sine=frequency=440:duration={duration}', str(source)], check=True)
    result = video.voice_sample(str(source))
    assert result
    info = video.probe(result)
    assert info['seconds'] == pytest.approx(expected, abs=0.15)
