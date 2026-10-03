from pathlib import Path
import math

OUT_FILE = Path("vef214_fm_scale_nonlinear.svg")

# ============================================
# GEOMETRY
# ============================================

WIDTH_MM = 135.0
HEIGHT_MM = 7.0

SCALE_LEN_MM = 135.0
MM_PER_UNIT = SCALE_LEN_MM / 10.0

# ============================================
# CALIBRATION
# ============================================

A = 0.0857142857
B = -3.5
C = 112.9571429

# ============================================
# MATH
# ============================================

def freq_from_pos(x: float) -> float:
    return A * x * x + B * x + C


def pos_from_freq(y: float) -> float:
    disc = B * B - 4 * A * (C - y)

    if disc < 0 and disc > -1e-12:
        disc = 0.0
    if disc < 0:
        raise ValueError(f"Frequency {y} outside range")

    return (-B - math.sqrt(disc)) / (2 * A)


def mm_from_freq(y: float) -> float:
    return pos_from_freq(y) * MM_PER_UNIT


def fmt_freq(y: float) -> str:
    # If value differs from integer by less than 0.05, treat as integer
    if abs(y - round(y)) < 0.05:
        return str(int(round(y)))
    # Otherwise show one decimal place
    return f"{y:.1f}"

# ============================================
# SVG
# ============================================


def build_svg() -> str:

    f_left = freq_from_pos(0.0)
    f_right = freq_from_pos(10.0)

    parts = []

    parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg"
     width="{WIDTH_MM}mm"
     height="{HEIGHT_MM}mm"
     viewBox="0 0 {WIDTH_MM} {HEIGHT_MM}">

  <defs>
    <style>

      .base {{
        stroke:black;
        stroke-width:0.18;
        fill:none;
      }}

      .major {{
        stroke:black;
        stroke-width:0.18;
      }}

      .mid {{
        stroke:black;
        stroke-width:0.12;
      }}

      .txt-top {{
        font-family: Arial, Helvetica, sans-serif;
        font-size: 2.2px;
        fill: black;
      }}

    </style>
  </defs>

''')

    # ============================================
    # BASELINE
    # ============================================

    parts.append(
        f'<line class="base" x1="0" y1="6" x2="{WIDTH_MM}" y2="6" />\n'
    )

    # ============================================
    # BOTTOM LINEAR SCALE (COMPRESSED)
    # ============================================

    for i in range(11):
        x = i * MM_PER_UNIT

        parts.append(
            f'<line class="major" x1="{x:.2f}" y1="6.2" x2="{x:.2f}" y2="7.2" />\n'
        )

    for i in range(10):
        x = i * MM_PER_UNIT + MM_PER_UNIT / 2.0

        parts.append(
            f'<line class="mid" x1="{x:.2f}" y1="6.7" x2="{x:.2f}" y2="7.2" />\n'
        )

    # ============================================
    # TOP NONLINEAR SCALE
    # ============================================

    start_tenths = int(math.ceil(f_right * 10))
    end_tenths = int(math.floor(f_left * 10))

    SCALE_Y_TOP = 2.6
    SCALE_Y_BOTTOM = 5.2

    for t in range(start_tenths, end_tenths + 1):

        freq = t / 10.0

        # ✔ correct filter skip point
        if abs(freq - 113.0) < 1e-6:
            continue

        try:
            x_mm = mm_from_freq(freq)
        except ValueError:
            continue

        if not (0.0 <= x_mm <= WIDTH_MM):
            continue

        if abs(freq - round(freq)) < 1e-9:
            y1, y2 = SCALE_Y_TOP, SCALE_Y_BOTTOM
        elif abs(freq * 2 - round(freq * 2)) < 1e-9:
            y1, y2 = 3.0, SCALE_Y_BOTTOM
        else:
            y1, y2 = 4.0, SCALE_Y_BOTTOM

        parts.append(
            f'<line class="major" x1="{x_mm:.2f}" y1="{y1:.2f}" x2="{x_mm:.2f}" y2="{y2:.2f}" />\n'
        )

    # ============================================
    # TOP FM LABELS
    # ============================================

    parts.append(
        f'<text class="txt-top" x="2.0" y="2" text-anchor="start">{fmt_freq(f_left)}</text>\n'
    )

    parts.append(
        f'<text class="txt-top" x="{WIDTH_MM - 2.0:.2f}" y="2" text-anchor="end">{fmt_freq(f_right)}</text>\n'
    )

    for f_label in range(math.ceil(f_right) + 1, math.floor(f_left)):

        freq = float(f_label)

        # ✔ fix: actually works now
        if abs(freq - 113.0) < 1e-6:
            continue

        try:
            x_mm = mm_from_freq(freq)
        except ValueError:
            continue

        if not (0.0 <= x_mm <= WIDTH_MM):
            continue

        parts.append(
            f'<text class="txt-top" x="{x_mm:.2f}" y="2" text-anchor="middle">{f_label}</text>\n'
        )

    parts.append("</svg>\n")

    return "".join(parts)


# ============================================
# MAIN
# ============================================

def main():
    OUT_FILE.write_text(build_svg(), encoding="utf-8")
    print(f"Saved: {OUT_FILE.resolve()}")



main()