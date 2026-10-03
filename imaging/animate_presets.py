"""Named motion presets for animating a single photo (generate_video with one
image attached -- video.pick_mode currently routes that through ref2va; see
its docstring for why the FL2VA-checkpoint i2va path is disabled). No
storyboard needed: each prompt describes a short, self-contained shot that
keeps the subject and setting from the photo and adds only camera/subject
motion, mirroring style_presets.py's "keep the content, change one axis"
shape.
"""

ANIMATE_PRESETS: dict[str, dict] = {
    "push_in": {
        "label_ru": "🎥 Наезд камеры", "label_en": "🎥 Camera push-in",
        "prompt": ("slow, smooth cinematic push-in on the subject, everything else "
                   "stays exactly as in the photo, subtle natural ambient motion "
                   "only, no cuts"),
    },
    "wind": {
        "label_ru": "🍃 Ветер", "label_en": "🍃 Wind",
        "prompt": ("a gentle breeze moves the subject's hair and any loose fabric, "
                   "camera stays still, everything else stays exactly as in the "
                   "photo"),
    },
    "smile_blink": {
        "label_ru": "😊 Улыбка и взгляд", "label_en": "😊 Smile & blink",
        "prompt": ("the subject blinks naturally and breaks into a warm smile, "
                   "head and body stay still, camera stays still, everything else "
                   "stays exactly as in the photo"),
    },
    "wave": {
        "label_ru": "👋 Машет рукой", "label_en": "👋 Waving hello",
        "prompt": ("the subject raises a hand and waves hello at the camera with a "
                   "friendly smile, camera stays still, everything else stays "
                   "exactly as in the photo"),
    },
    "orbit": {
        "label_ru": "🔄 Облёт камеры", "label_en": "🔄 Camera orbit",
        "prompt": ("slow cinematic camera orbit a few degrees around the subject, "
                   "the subject stays still, everything else stays exactly as in "
                   "the photo"),
    },
    "walk": {
        "label_ru": "🚶 Идёт к камере", "label_en": "🚶 Walking closer",
        "prompt": ("the subject takes a few steps forward toward the camera "
                   "naturally, camera stays still, everything else stays exactly "
                   "as in the photo"),
    },
}

ANIMATE_PRESET_ORDER = ("push_in", "wind", "smile_blink", "wave", "orbit", "walk")


def preset_prompt(key: str) -> str:
    """The generate_video description for `key`, or "" if not a known preset."""
    p = ANIMATE_PRESETS.get(key)
    return p["prompt"] if p else ""


def preset_label(key: str, lang: str = "ru") -> str:
    p = ANIMATE_PRESETS.get(key)
    if not p:
        return key
    return p["label_ru"] if lang == "ru" else p["label_en"]
