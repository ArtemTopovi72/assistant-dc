"""Helpers preinstalled in the run_code sandbox image (import assistant_tools).

Measured 2026-09-14: asked to drop near-duplicate photos, the model wrote an
exact-bytes "hash" three times in a row even after being told to use a
perceptual hash -- so the working implementation is shipped as a function.
And a perceptual hash would not have worked either: re-shots of the same
bridge from another angle hash as far apart as unrelated photos. "Duplicate"
to a person means the same scene, which is a keypoint-matching question.
"""
from __future__ import annotations

import math
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def list_images(folder=".", recursive=True):
    """Every image under `folder`, sorted, hidden folders skipped."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    out = []
    for p in sorted(it):
        if any(part.startswith(".") for part in p.relative_to(folder).parts):
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def _open(path):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def sharpness(path) -> float:
    """Variance of the Laplacian on a 1024-px copy: higher = sharper focus."""
    import numpy as np
    im = _open(path).convert("L")
    im.thumbnail((1024, 1024))
    a = np.asarray(im, dtype="float64")
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    return float(lap.var())


def phash(path):
    import imagehash
    return imagehash.phash(_open(path))


def _orb_features(path, size: int = 800):
    import cv2, numpy as np
    im = _open(path).convert("L")
    im.thumbnail((size, size))
    orb = cv2.ORB_create(1500)
    return orb.detectAndCompute(np.asarray(im), None)


def same_scene_score(feats_a, feats_b) -> int:
    """RANSAC-consistent keypoint matches between two photos.

    Measured on a holiday folder: the same bridge re-shot from another angle
    scores 39-523, unrelated photos 0-11. Perceptual hashes cannot do this --
    those same re-shots hashed as far apart as strangers (pHash 14-38).
    """
    import cv2, numpy as np
    ka, da = feats_a
    kb, db = feats_b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = [t for t in bf.knnMatch(da, db, k=2) if len(t) == 2]
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([ka[g.queryIdx].pt for g in good])
    dst = np.float32([kb[g.trainIdx].pt for g in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def group_near_duplicates(paths, threshold: int = 25, window: int = 16):
    """Groups of photos showing the SAME SCENE (re-shots, bursts, other angles).

    Returns a list of lists; singletons included, so every input path is in
    exactly one group. `threshold` is the number of geometrically consistent
    keypoint matches (see same_scene_score); 25 separates re-shots from
    unrelated photos with a wide margin on real holiday folders.

    Every pair is compared when the set is small. A whole folder is not:
    173 photos are 14.9k ORB matches, ~30 s on a desktop and past the
    run_code timeout inside the container (live 2026-09-14, after which the
    model wrote its own pHash). Re-shots are shot together, so for a big set
    only photos within `window` positions of each other in name order are
    compared -- 2.7k matches for those 173 -- and this also drops the one
    repetitive-texture photo that scored 26-34 against strangers far away.
    """
    paths = [Path(p) for p in paths]
    n = len(paths)
    feats = [_orb_features(p) for p in paths]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n <= 40 or window <= 0:
        pairs = ((i, j) for i in range(n) for j in range(i))
    else:
        order = sorted(range(n), key=lambda i: (str(paths[i].parent).lower(), paths[i].name.lower()))
        pairs = ((order[a], order[b]) for a in range(n)
                 for b in range(max(0, a - window), a))
    for i, j in pairs:
        if find(i) != find(j) and same_scene_score(feats[i], feats[j]) >= threshold:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [[paths[i] for i in g] for g in groups.values()]


def dedupe(paths, threshold: int = 25, verbose: bool = True):
    """Keep ONE photo per same-scene group: the sharpest. Returns the kept paths.

    Prints every group and what was kept, so the output proves what happened.
    """
    paths = [Path(p) for p in paths]
    kept = []
    for g in group_near_duplicates(paths, threshold):
        if len(g) == 1:
            kept.append(g[0]); continue
        scored = sorted(((sharpness(p), p) for p in g), reverse=True)
        best = scored[0][1]
        kept.append(best)
        if verbose:
            print(f"duplicate group ({len(g)}): " + ", ".join(p.name for p in g)
                  + f" -> kept {best.name} (sharpness {scored[0][0]:.0f})")
    if verbose:
        print(f"dedupe: {len(paths)} photos -> {len(kept)} unique")
    return kept


def make_collage(paths, out="collage.jpg", cols: int | None = None, cell: int = 800,
                 gap: int = 8, background=(255, 255, 255)):
    """Tile photos into a grid (aspect kept, letterboxed) and save it. Returns `out`."""
    from PIL import Image
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no photos to collage")
    n = len(paths)
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * cell + (cols + 1) * gap, rows * cell + (rows + 1) * gap), background)
    for i, p in enumerate(paths):
        im = _open(p)
        im.thumbnail((cell, cell))
        x = gap + (i % cols) * (cell + gap) + (cell - im.width) // 2
        y = gap + (i // cols) * (cell + gap) + (cell - im.height) // 2
        sheet.paste(im, (x, y))
    sheet.save(out, quality=90)
    print(f"collage: {n} photos, {cols}x{rows} -> {out}")
    return out
