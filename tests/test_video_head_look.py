"""A talking-head part is redrawn when its light leaves the still's, as when its eyes do.

Run 16 (2026-10-20): the rewrite gave the soundscape «the rhythmic buzz of flickering lights»
and part 5 flashed for ten seconds, the walls at 1.8-2.9 of the still's light; parts 1 and 3 sat
in a milky haze at 1.3-1.6. The eye checks passed all of it, and the first 3 s of part 1 had
the eyes shut -- which the template and the brightness both read as the still's shadowed eyes.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "media"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import cv2  # noqa: E402
import video  # noqa: E402


def _scene(level: float) -> np.ndarray:
    img = np.full((120, 160, 3), 40, np.uint8)
    img[:, :40] = img[:, -40:] = int(level)          # the walls either side
    img[30:90, 60:100] = 150                          # the face
    return img


def _clip(path, levels, fps=24):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for lv in levels:
        vw.write(_scene(lv))
    vw.release()
    return str(path)


def test_steady_light_passes_and_a_flashing_or_hazy_part_does_not(tmp_path):
    still = str(tmp_path / "still.png")
    cv2.imwrite(still, _scene(42))
    steady = _clip(tmp_path / "steady.mp4", [42 + (i % 3) for i in range(72)])
    flashing = _clip(tmp_path / "flash.mp4", [42] * 12 + [110 if i % 5 < 2 else 80 for i in range(60)])
    hazy = _clip(tmp_path / "haze.mp4", [42] * 12 + [64] * 60)
    assert video.look_drift(still, steady) <= 0
    assert video.look_drift(still, flashing) > 0
    assert video.look_drift(still, hazy) > 0


def test_a_hazy_take_kept_as_the_best_is_pulled_back_to_the_stills_light(tmp_path):
    """Run 17: the best of three hazy takes went in as it was (2026-10-20)."""
    still = str(tmp_path / "still.png")
    cv2.imwrite(still, _scene(42))
    hazy = _clip(tmp_path / "haze.mp4", [42] * 6 + [42 + 28 * i / 24 for i in range(24)] + [70] * 42)
    assert video.look_drift(still, hazy) > 0
    pulled = video.match_look(still, hazy)
    assert pulled and video.look_drift(still, pulled) <= 0
    assert video.match_look(still, str(tmp_path / "none.mp4")) is None


def test_a_broken_read_never_blocks_a_render(tmp_path):
    assert video.look_drift(str(tmp_path / "none.png"), str(tmp_path / "none.mp4")) is None
    assert video.eye_shut(np.zeros((10, 10, 3), np.uint8)) is None      # no face: no verdict


def test_the_head_lock_asks_for_open_eyes_and_steady_light():
    lock = video._HEAD_LOCK_SUFFIX
    assert "never closed" in lock and "no flicker" in lock


def test_a_push_in_to_the_face_is_not_the_stills_framing_whatever_its_light(tmp_path):
    """Run 18: two parts of six pushed in until the face filled the frame (2026-10-20)."""
    still_img = _scene(42)
    still_img[95:110, 20:140] = 200                 # a shelf, so the frame has edges to match
    still = str(tmp_path / "still.png")
    cv2.imwrite(still, still_img)
    vw = cv2.VideoWriter(str(tmp_path / "push.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 120))
    for k in range(48):
        z = 1 + min(1.0, k / 12)                    # zooms to 2x in half a second
        vw.write(cv2.resize(still_img, None, fx=z, fy=z)[:120, :160] if z > 1 else still_img)
    vw.release()
    lit = cv2.VideoWriter(str(tmp_path / "lit.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 120))
    for k in range(48):
        lit.write(cv2.convertScaleAbs(still_img, alpha=1.4, beta=20))
    lit.release()
    assert video.framing_kept(still, str(tmp_path / "push.mp4")) is False
    assert video.framing_kept(still, str(tmp_path / "lit.mp4")) is True


def test_a_talking_heads_prompt_never_invites_a_camera_move():
    """«any camera move or reframing happens only later» was in every head prompt of run 18."""
    assert "only later" not in video._HEAD_START_LOCK_SUFFIX
    assert "never a closer shot" in video._HEAD_LOCK_SUFFIX
