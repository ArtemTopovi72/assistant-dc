"""Objective identity + edit-containment metrics for image-edit validation.

- **Identity**: YuNet face detector + SFace recognizer (OpenCV Zoo ONNX models).
  Cosine similarity in SFace embedding space; SFace's "same person" threshold is
  **0.363** (cosine). 1.0 = identical crop.
- **Containment**: fraction of pixels that changed *outside* an allowed edit mask.
  For a localized edit this should be ~0 (only the masked region may change).

No network at runtime; models live in models/face/. Degrades gracefully (returns
None) if a face isn't found so callers can branch.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("assistant.identity")

_MODELS = Path(__file__).resolve().parents[1] / "models" / "face"
_YUNET = str(_MODELS / "face_detection_yunet_2023mar.onnx")
_SFACE = str(_MODELS / "face_recognition_sface_2021dec.onnx")
SFACE_SAME_PERSON_COSINE = 0.363   # OpenCV's documented SFace threshold

_recognizer = None


def _rec():
    global _recognizer
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(_SFACE, "")
    return _recognizer


def _largest_face(img):
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(_YUNET, "", (w, h), score_threshold=0.6)
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    # faces[:, :4] = x,y,w,h ; pick the largest
    return max(faces, key=lambda f: f[2] * f[3])


def face_count(path: str, *, score_threshold: float = 0.7) -> Optional[int]:
    """Number of faces YuNet detects, or None if the image can't be read /
    detector is unavailable. Used as a cheap deterministic gate against the
    duplicated-person render artifact (an edit must not ADD faces)."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(1.0, 1280.0 / max(w, h))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]
        det = cv2.FaceDetectorYN.create(_YUNET, "", (w, h),
                                        score_threshold=score_threshold)
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        return 0 if faces is None else int(len(faces))
    except Exception:
        return None


def face_embedding(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    face = _largest_face(img)
    if face is None:
        return None
    aligned = _rec().alignCrop(img, face.reshape(1, -1))
    return _rec().feature(aligned)


def identity_cosine(path_a: str, path_b: str) -> Optional[float]:
    """SFace cosine similarity between the largest face in each image.
    >0.363 = same person; ~1.0 = unchanged face. None if a face is missing."""
    fa, fb = face_embedding(path_a), face_embedding(path_b)
    if fa is None or fb is None:
        return None
    return float(_rec().match(fa, fb, cv2.FaceRecognizerSF_FR_COSINE))


def changed_fraction(path_a: str, path_b: str, *, mask: Optional[np.ndarray] = None,
                     thresh: int = 18) -> dict:
    """Fraction of pixels that differ between A and B.

    Returns {'overall', 'outside_mask', 'inside_mask'} as fractions in [0,1].
    `mask` (uint8, >0 = allowed edit region) is resized to A. 'outside_mask' is
    the edit-containment metric: ~0 means the edit stayed inside the region.
    """
    a = cv2.imread(path_a)
    b = cv2.imread(path_b)
    if a is None or b is None:
        return {}
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    changed = diff > thresh
    out = {"overall": float(changed.mean())}
    if mask is not None:
        m = cv2.resize(mask, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        outside = changed & ~m
        inside = changed & m
        out["outside_mask"] = float(outside.sum() / max(1, (~m).sum()))
        out["inside_mask"] = float(inside.sum() / max(1, m.sum()))
    return out


def face_region_change(path_a: str, path_b: str, *, thresh: int = 18, pad: float = 0.15):
    """Fraction of pixels changed inside A's (padded) face box. ~0 = face left
    untouched (containment success); high = the face was regenerated."""
    a = cv2.imread(path_a); b = cv2.imread(path_b)
    if a is None or b is None:
        return None
    f = _largest_face(a)
    if f is None:
        return None
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    H, W = a.shape[:2]
    x, y, fw, fh = f[:4]
    x0 = max(0, int(x - pad * fw)); y0 = max(0, int(y - pad * fh))
    x1 = min(W, int(x + fw + pad * fw)); y1 = min(H, int(y + fh + pad * fh))
    ca = cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    cb = cv2.cvtColor(b[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float((cv2.absdiff(ca, cb) > thresh).mean())


def face_bbox_frac(path: str):
    """Largest face as (cx,cy,w,h) fractions of the frame, or None."""
    img = cv2.imread(path)
    if img is None:
        return None
    f = _largest_face(img)
    if f is None:
        return None
    h, w = img.shape[:2]
    x, y, fw, fh = f[:4]
    return ((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h)


def face_boxes(path: str, pad: float = 0.2, *, max_faces: int = 12,
               score_threshold: float = 0.6) -> list:
    """EVERY detected face as a padded pixel box (x0, y0, x1, y1), largest first.

    A group photo is the normal case, not an edge case: protecting only the
    biggest face left the two policemen in a three-person photo to be
    hallucinated by the super-resolver while the man in front stayed faithful.
    Detection runs on a downscaled copy for speed/robustness on large images,
    then every box is scaled back to the source resolution.
    """
    img = cv2.imread(path)
    if img is None:
        return []
    H, W = img.shape[:2]
    scale = min(1.0, 1600.0 / max(H, W))
    small = cv2.resize(img, (int(W * scale), int(H * scale))) if scale < 1.0 else img
    h, w = small.shape[:2]
    try:
        det = cv2.FaceDetectorYN.create(_YUNET, "", (w, h),
                                        score_threshold=score_threshold)
        det.setInputSize((w, h))
        _, faces = det.detect(small)
    except Exception:
        logger.warning("face_boxes: detector unavailable", exc_info=True)
        return []
    if faces is None or len(faces) == 0:
        return []
    out = []
    for f in sorted(faces, key=lambda f: -(f[2] * f[3]))[:max_faces]:
        x, y, fw, fh = (v / scale for v in f[:4])
        x0 = max(0, int(x - pad * fw)); y0 = max(0, int(y - pad * fh))
        x1 = min(W, int(x + fw + pad * fw)); y1 = min(H, int(y + fh + pad * fh))
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out


def face_box(path: str, pad: float = 0.2):
    """Largest face as a padded pixel box (x0, y0, x1, y1), or None.

    Used to PROTECT the face from non-face edits: the box is subtracted from the
    edit mask so the inpaint/composite can never alter facial pixels. Callers
    that must cover a GROUP want face_boxes() instead — this one is deliberately
    the single dominant face (mask subtraction for a portrait-style edit).
    """
    boxes = face_boxes(path, pad)
    return boxes[0] if boxes else None
