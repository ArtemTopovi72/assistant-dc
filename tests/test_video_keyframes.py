"""Key moments by frame difference (video_look.key_times).

The user's case: someone looks into the camera, swings it (blur) onto a
step with «67» painted on it, then holds. «Every 5 s» could land in the
swing; the key-moment pass must give the face, the step, and nothing from
the smear in between.
"""
import os, sys, subprocess, shutil
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import video_look as V


def _still(path: Path, seed: int, size=(320, 240)) -> Path:
    """A sharp, textured still (real footage is never a flat colour)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _clip(tmp: Path) -> Path:
    """0-4 s a sharp 'face' still, 4-6 s a heavily blurred moving pan, 6-12 s a
    sharp 'step' still."""
    a, b = _still(tmp / "face.png", 1), _still(tmp / "step.png", 2)
    out = tmp / "keys.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-loop", "1", "-t", "4", "-i", str(a),
                    "-f", "lavfi", "-i", "testsrc=s=320x240:d=2:r=25",
                    "-loop", "1", "-t", "6", "-i", str(b),
                    "-filter_complex",
                    "[0]format=yuv420p,fps=25[f];[1]boxblur=20:2,format=yuv420p[m];"
                    "[2]format=yuv420p,fps=25[s];[f][m][s]concat=n=3:v=1:a=0[v]",
                    "-map", "[v]", "-r", "25", str(out)], check=True, timeout=120)
    return out


def test_key_times_face_then_step(tmp_path: Path):
    clip = _clip(tmp_path)
    frames = V.dense_pass(str(clip), tmp_path / "dense")
    ts = V.key_times(frames, max_frames=6)
    assert ts[0] < 4.0, ts                        # the 'face' shot
    assert any(t >= 6.0 for t in ts), ts          # the 'step' shot
    # nothing picked from inside the 2-second swing (4-6 s) -- the boundary
    # frames are the swing, and blur loses to sharp frames either side
    assert not [t for t in ts if 4.4 <= t <= 5.6], ts
    assert len(ts) <= 6


def test_static_clip_gets_anchors_not_dozens():
    # 60 identical tiny frames at 2 fps = a 30 s clip that never changes
    fr = [(i / 2, np.full((54, 96), 100.0, dtype=np.float32)) for i in range(60)]
    ts = V.key_times(fr, max_frames=12)
    assert 2 <= len(ts) <= 12 and ts[0] <= 1.0 and ts[-1] >= 20.0, ts


def test_handheld_noise_is_not_a_cut_every_frame():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, (54, 96)).astype(np.float32)
    # 68 frames of the same picture with hand-held jitter (noise floor ~30)
    fr = [(i / 2, np.clip(base + rng.normal(0, 40, base.shape), 0, 255).astype(np.float32))
          for i in range(68)]
    ts = V.key_times(fr, max_frames=12)
    assert len(ts) <= 12, ts
    gaps = np.diff(ts)
    assert gaps.min() >= 1.0, ts                  # never two picks inside a second


if __name__ == "__main__":
    import tempfile
    t = Path(tempfile.mkdtemp())
    try:
        test_key_times_face_then_step(t)
        test_static_clip_gets_anchors_not_dozens()
        test_handheld_noise_is_not_a_cut_every_frame()
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("3/3 passed")
