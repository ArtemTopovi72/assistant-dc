import cv2
import numpy as np

import video


def test_detects_a_new_sustained_bright_area(tmp_path):
    reference = np.full((144, 320, 3), 30, dtype=np.uint8)
    image = str(tmp_path / 'reference.png')
    cv2.imwrite(image, reference)

    def clip(name, flash):
        path = str(tmp_path / name)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 4, (320, 144))
        assert writer.isOpened()
        for i in range(16):
            frame = reference.copy()
            if flash and 4 <= i < 14:
                cv2.circle(frame, (235, 75), 15, (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        return path

    assert video.new_bright_region(clip('flash.mp4', True), image) is True
    assert video.new_bright_region(clip('clean.mp4', False), image) is False
