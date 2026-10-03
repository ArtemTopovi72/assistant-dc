"""Measured exposure of a render, for judges that cannot be trusted on it.

The house vision model misjudges exposure in one direction, consistently: a
clearly lit orange cat on a sunlit sill is "a black silhouette", the wooden
sill under it "a large black bar", an ordinary interior "mostly black" (bench
2026-08-29; live 2026-09-12, three renders in a row at 5/10, 4/10, 5/10 for the
same imagined darkness). Asked to simply describe the same picture it names
the cat, the window and the snow correctly -- and still calls it dark.

Brightness is one of the few things a judge says that can be MEASURED, so it
is measured here, once, and both judges (the layout critic in draw_agent and
the outer evaluate_image loop) use the same numbers: the picture is told to
the judge as evidence before it answers, and a darkness complaint that the
pixels contradict is dropped after.
"""
from __future__ import annotations

import re
from typing import Optional

# Below these the picture is genuinely dark and the complaint stands.
MEAN_FLOOR = 55.0      # mean luminance, 0-255
P95_FLOOR = 140.0      # 95th percentile

DARK_RE = re.compile(
    r"too dark|very dark|extremely dark|underexpos|under-expos|mostly black|"
    r"poorly lit|badly lit|barely visible|hard to see|low[- ]light|"
    r"heavy shadows|dark shadows|silhouett|black bars?|letterbox|"
    r"obscured by (?:a )?(?:large )?(?:black|dark)|lost (?:to|in) shadow|"
    r"deep blacks|black silhouette|darkness|crushed (?:in|to|into) (?:the )?blacks?|"
    r"too dim|dimly lit|extremely dim|"
    r"(?:almost |nearly |mostly )?(?:entirely|completely|all) (?:black|dark)|"
    r"lost (?:most|all) (?:of )?(?:its |their |the )?(?:texture|detail)", re.IGNORECASE)

# The judge says the WHOLE picture is black. About a picture that measures as
# normally exposed, that is not a complaint, it is proof the judge did not see
# the picture -- and then nothing else it says about it can be trusted either.
BLIND_RE = re.compile(
    r"(?:image|picture|scene|render|frame|it) (?:is|appears|looks) (?:almost |nearly |mostly )?"
    r"(?:entirely |completely |all |very |extremely |too )?(?:black|dark|underexposed)|"
    r"(?:almost |nearly )?(?:entirely|completely) (?:black|dark)|"
    r"(?:extremely|severely|heavily) (?:dark|underexposed)", re.IGNORECASE)


def judge_is_blind(problems, image_path: str) -> bool:
    """True when the judge called a normally exposed picture black overall."""
    if not normally_exposed(measure(image_path)):
        return False
    return any(BLIND_RE.search(str(p or "")) for p in (problems or []))


def measure(image_path: str) -> Optional[tuple]:
    """(mean, 95th percentile) luminance of the picture, or None."""
    try:
        import numpy as _np
        from PIL import Image as _Image
        a = _np.asarray(_Image.open(image_path).convert("L"), dtype=float)
        return float(a.mean()), float(_np.percentile(a, 95))
    except Exception:
        return None


def normally_exposed(m: Optional[tuple]) -> bool:
    """True when the pixels say the picture is ordinarily lit."""
    return bool(m) and m[0] >= MEAN_FLOOR and m[1] >= P95_FLOOR


def evidence(image_path: str) -> str:
    """A line for the judge's prompt, or "" when the picture really is dark."""
    m = measure(image_path)
    if not normally_exposed(m):
        return ""
    return ("Measured exposure of this picture: mean luminance %d/255, 95th percentile "
            "%d/255 -- it is normally exposed. Do not report it as dark, underexposed, "
            "a silhouette, or having black bars; those would be errors of perception, "
            "not of the picture." % (m[0], m[1]))


# A complaint that survives having its darkness sentences removed: the subject
# is not there, is the wrong thing, is doubled, broken or cut off.
_SUBSTANTIVE_RE = re.compile(
    r"\b(?:absent|missing|not present|no [a-z]+ (?:is|are) (?:visible|present|shown)|"
    r"instead of|wrong|extra|duplicate|duplicated|second [a-z]+|another [a-z]+|fused|melted|merged|"
    r"deformed|warped|cropped|cut off|floating|pasted|blurry|out of focus|"
    r"appears twice|two (?:heads|faces))\b", re.IGNORECASE)


def is_dark_complaint(text: str) -> bool:
    """True when the complaint is ONLY about darkness.

    Sentences that mention darkness are set aside; if what is left still names
    an absent, wrong, doubled or broken subject, the complaint stands.
    """
    t = str(text or "")
    if not DARK_RE.search(t):
        return False
    rest = [s for s in re.split(r"(?<=[.;!?])\s+", t) if s.strip() and not DARK_RE.search(s)]
    return not any(_SUBSTANTIVE_RE.search(s) for s in rest)
