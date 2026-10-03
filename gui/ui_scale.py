"""Resolution- and DPI-aware UI scaling for the PyQt5 desktop app.

The app is styled almost entirely through one QSS string plus a handful of inline
styles and fixed widget sizes, all authored against a 1920x1080 @ 100% baseline.
This module turns that baseline into a single scalable design system:

  * Qt high-DPI is enabled (run_gui sets the attributes BEFORE QApplication), so
    Windows display scaling (100..300%) is honoured automatically by Qt — px in QSS
    and point sizes are multiplied by the OS device-pixel-ratio.

  * On top of that we apply an *application* scale factor for the dimension Qt's DPI
    ratio does NOT cover: physical resolution + viewing distance. A 4K panel set to
    100% Windows scaling (common on TVs) reports devicePixelRatio 1.0, so Qt does not
    enlarge anything and the UI is physically tiny. ``auto_factor`` targets a constant
    *total* magnification per physical resolution and divides by what Qt already does,
    so 4K lands near 2x whether Windows is at 100%, 150% or 200%.

  * ``TV Mode`` adds a further multiplier for couch viewing distance.

Everything funnels through ``px()`` / ``pt()`` / ``scale_style()`` so a single factor
rescales fonts, paddings, fixed sizes and inline styles consistently.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("assistant.ui_scale")

_SETTINGS_FILE = Path(__file__).resolve().parents[1] / "ui_settings.json"

# Discrete scale steps offered in Settings (mirrors Windows' own ladder).
SCALE_STEPS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
SCALE_OPTIONS = ["Auto (recommended)", "100%", "125%", "150%", "175%", "200%", "250%", "300%"]

# Density presets (overall size multiplier on top of the chosen UI scale).
DENSITY = {"compact": 0.92, "comfortable": 1.0, "tv": 1.28}
DENSITY_OPTIONS = ["Compact", "Comfortable", "TV Mode"]
_DENSITY_KEYS = ["compact", "comfortable", "tv"]

# Font Scale presets — multiply ONLY fonts, on top of everything else.
FONT_OPTIONS = ["90%", "100%", "110%", "125%", "150%"]
_FONT_STEPS = [0.9, 1.0, 1.1, 1.25, 1.5]

# Interface language: code -> the name shown in Settings (in its own language).
LANGUAGES = {"en": "English", "ru": "Русский"}

_state = {
    "scale_mode": "auto",       # "auto" or one of "100".."300"
    "density": "comfortable",   # compact | comfortable | tv
    "font_scale": 1.0,          # extra font-only multiplier
    "language": "en",           # interface language (gui_i18n): en | ru
}
# Resolved at startup once the screen is known. 1.0 until apply_screen() runs.
_auto = 1.0
_effective = 1.0       # layout/size scale (no font multiplier)
_font_mult = 1.0       # font-only multiplier (font_scale)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load() -> None:
    try:
        data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if data.get("scale_mode"):
                _state["scale_mode"] = str(data["scale_mode"])
            # migrate the old boolean tv_mode → density
            if data.get("density") in DENSITY:
                _state["density"] = data["density"]
            elif data.get("tv_mode"):
                _state["density"] = "tv"
            if data.get("language") in LANGUAGES:
                _state["language"] = data["language"]
            try:
                _state["font_scale"] = float(data.get("font_scale", 1.0))
            except Exception:
                _state["font_scale"] = 1.0
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("ui_scale: could not load settings: %s", exc)


def save() -> None:
    try:
        _SETTINGS_FILE.write_text(json.dumps(_state, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("ui_scale: could not save settings: %s", exc)


# --------------------------------------------------------------------------- #
# Scale resolution
# --------------------------------------------------------------------------- #
def _snap(factor: float) -> float:
    """Snap an arbitrary factor to the nearest discrete step."""
    return min(SCALE_STEPS, key=lambda s: abs(s - factor))


def auto_factor(screen) -> float:
    """Compute the recommended application scale for ``screen`` (a QScreen).

    Strategy: pick a *target total magnification* from the physical resolution,
    then divide by the magnification Qt already applies via the OS DPI ratio, so
    the on-screen size is consistent regardless of the Windows scaling setting.
    """
    try:
        geo = screen.geometry()
        dpr = float(screen.devicePixelRatio()) or 1.0
        # geometry() is in logical px under high-DPI; physical = logical * dpr
        phys_short = int(min(geo.width(), geo.height()) * dpr)
    except Exception:
        return 1.0

    # Target TOTAL magnification (app * dpr) keyed to physical short side.
    if phys_short <= 1100:        # 1080p and below
        target = 1.0
    elif phys_short <= 1500:      # 1440p / QHD
        target = 1.25
    elif phys_short <= 1700:      # in-between / ultrawide QHD
        target = 1.5
    else:                          # 2160p (4K) and above
        target = 2.0

    # Larger physical panels (TVs) viewed further away want a touch more.
    try:
        phys = screen.physicalSize()  # millimetres
        diag_in = ((phys.width() ** 2 + phys.height() ** 2) ** 0.5) / 25.4
        if diag_in >= 40:          # television territory
            target *= 1.15
    except Exception:
        pass

    app_factor = target / dpr
    return _snap(max(1.0, min(3.0, app_factor)))


def apply_screen(screen) -> None:
    """Resolve the auto factor for the active screen and recompute effective scale."""
    global _auto
    _auto = auto_factor(screen) if screen is not None else 1.0
    _recompute()


def _recompute() -> None:
    global _effective, _font_mult
    mode = _state["scale_mode"]
    if mode == "auto":
        base = _auto
    else:
        try:
            base = max(1.0, min(3.0, int(str(mode).rstrip("%")) / 100.0))
        except Exception:
            base = _auto
    base *= DENSITY.get(_state["density"], 1.0)
    _effective = round(max(0.8, min(4.0, base)), 3)
    _font_mult = round(max(0.6, min(2.0, float(_state.get("font_scale", 1.0)))), 3)


# --------------------------------------------------------------------------- #
# Public knobs (used by the Settings dialog)
# --------------------------------------------------------------------------- #
def set_mode(scale_mode: str, density: str = "comfortable", font_scale: float = 1.0) -> None:
    _state["scale_mode"] = scale_mode
    if density in DENSITY:
        _state["density"] = density
    _state["font_scale"] = float(font_scale)
    _recompute()


def language() -> str:
    return _state["language"]


def set_language(lang: str) -> None:
    if lang in LANGUAGES:
        _state["language"] = lang


def scale_mode() -> str:
    return _state["scale_mode"]


def density() -> str:
    return _state["density"]


def font_scale() -> float:
    return float(_state.get("font_scale", 1.0))


def tv_mode() -> bool:
    return _state["density"] == "tv"


def effective() -> float:
    return _effective


def auto_value() -> float:
    return _auto


def density_index() -> int:
    try:
        return _DENSITY_KEYS.index(_state["density"])
    except ValueError:
        return 1


def index_to_density(i: int) -> str:
    return _DENSITY_KEYS[i] if 0 <= i < len(_DENSITY_KEYS) else "comfortable"


def font_index() -> int:
    return min(range(len(_FONT_STEPS)), key=lambda i: abs(_FONT_STEPS[i] - font_scale()))


def index_to_font(i: int) -> float:
    return _FONT_STEPS[i] if 0 <= i < len(_FONT_STEPS) else 1.0


def mode_to_combo_index(mode: str) -> int:
    if mode == "auto":
        return 0
    try:
        step = int(str(mode).rstrip("%")) / 100.0
        return SCALE_STEPS.index(_snap(step)) + 1
    except Exception:
        return 0


def combo_index_to_mode(index: int) -> str:
    if index <= 0:
        return "auto"
    pct = int(round(SCALE_STEPS[index - 1] * 100))
    return str(pct)


# --------------------------------------------------------------------------- #
# Scaling primitives
# --------------------------------------------------------------------------- #
def px(n: float) -> int:
    """Scale a device-independent LAYOUT pixel size by the effective UI scale
    (does NOT include the font-only multiplier)."""
    return max(1, int(round(n * _effective)))


def fpx(n: float) -> int:
    """Scale a FONT pixel size: UI scale × font-scale multiplier."""
    return max(6, int(round(n * _effective * _font_mult)))


def pt(n: float) -> int:
    """Scale a font point size by UI scale × font-scale multiplier.

    Identical formula to ``fpx`` (both are "font units" — px and pt are treated
    the same way here); kept as a separate public name since callers use it to
    mean point sizes specifically."""
    return fpx(n)


_PX_RE = re.compile(r"(\d+)px")


def scale_style(qss: str) -> str:
    """Multiply every ``<n>px`` literal in an (inline) stylesheet by the scale."""
    return _PX_RE.sub(lambda m: f"{px(int(m.group(1)))}px", qss)
