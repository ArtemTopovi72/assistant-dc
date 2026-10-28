import cv2
import numpy as np
import pytest

from video import track_simple_motion, overlay_drift, OVERLAY_DRIFT_MAX


def _clip(path, positions, busy=False):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 180))
    assert out.isOpened()
    for x in positions:
        frame = np.full((180, 320, 3), (180, 90, 20), dtype=np.uint8)
        if busy:
            frame[:, :10] = (255, 255, 255)
            frame[:, -10:] = (0, 0, 0)
        cv2.circle(frame, (x, 90), 25, (10, 10, 230), -1)
        cv2.circle(frame, (x - 8, 80), 5, (120, 120, 255), -1)
        out.write(frame)
    out.release()


@pytest.mark.parametrize('positions,sign', [([70, 85, 110, 130, 160], 1),
                                             ([210, 195, 170, 150, 120], -1)])
def test_simple_object_motion_direction(tmp_path, positions, sign):
    path = tmp_path / 'clip.mp4'
    _clip(path, positions)
    result = track_simple_motion(str(path))
    assert result is not None
    assert sign * (result[-1][1] - result[0][1]) > 0.1


def test_busy_scene_is_left_to_visual_review(tmp_path):
    path = tmp_path / 'busy.mp4'
    _clip(path, [70, 85, 110, 130, 160], busy=True)
    assert track_simple_motion(str(path)) is None


def test_two_subjects_are_not_mistaken_for_a_single_track(tmp_path):
    path = tmp_path / 'two.mp4'
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 180))
    assert out.isOpened()
    for x in (65, 75, 85, 95, 105):
        frame = np.full((180, 320, 3), (180, 90, 20), dtype=np.uint8)
        cv2.circle(frame, (x, 70), 20, (10, 10, 230), -1)
        cv2.circle(frame, (250, 110), 20, (10, 200, 10), -1)
        out.write(frame)
    out.release()
    assert track_simple_motion(str(path)) is None


def test_background_colour_change_is_not_hidden_by_equal_brightness(tmp_path):
    path = tmp_path / 'colour_change.mp4'
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 180))
    assert out.isOpened()
    for i in range(20):
        colour = (100, 0, 0) if i < 10 else (0, 0, 39)
        out.write(np.full((180, 320, 3), colour, dtype=np.uint8))
    out.release()
    assert overlay_drift(str(path)) > OVERLAY_DRIFT_MAX
