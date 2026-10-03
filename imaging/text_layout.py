"""Text box geometry: the shape a string needs in order to be legible.

Extracted from draw_agent.py because ideogram.py needed two names from it and
was taking the whole layout-editing agent to get them — with a function-local
`from draw_agent import text_geometry, MIN_TEXT_H` written that way to dodge the
ideogram <-> draw_agent import cycle.

Nothing here knows about layouts, agents or renderers. It is arithmetic over a
string and a box, which is exactly why both modules can share it without either
depending on the other.

The numbers are empirical, not taste: a police car whose lettering box was
0.09 x 0.05 came back reading "AVCHKE" — six characters squeezed into a box
barely wider than it was tall, so the model painted letter-shaped strokes
instead of letters.
"""
import re


# --------------------------------------------------------------------------- #
# Ideogram renders legible text — when the box it is given can hold the string.
# A police car whose lettering box was 0.09 x 0.05 came back reading "AVCHKE":
# six characters squeezed into a box barely wider than it was tall, so the model
# painted letter-shaped strokes instead of letters. These numbers are the shape a
# string actually needs.
MIN_TEXT_H = 0.08        # below this the glyphs have too few pixels to form
CHAR_ASPECT = 0.6        # width of one block capital, in units of the box height
MAX_TEXT_CHARS = 24      # past this the model stops spelling and starts scribbling
SHAPE_TOLERANCE = 1.6    # how far off the natural aspect a box may be
TEXT_MATCH_OK = 0.8      # normalised similarity that counts as "it says the word"


def _norm_text(s: str) -> str:
    """Letters and digits only, lowercased — the form two spellings are compared in.
    Case, spacing and punctuation are the renderer's business, not a defect."""
    return re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", "", s or "").lower()


def lines_of(text: str) -> list:
    """The lines a string is laid out in. A single-line string is one line."""
    return [ln.strip() for ln in str(text or "").split("\n") if ln.strip()] or [""]


def wrap_text(text: str, max_chars: int) -> list:
    """Balanced word wrap into the fewest lines of at most `max_chars`.

    'Для гостей театра на Пролетарской' (34) at 18 -> ['Для гостей театра',
    'на Пролетарской']. A word longer than the limit stands on its own line;
    it is never split -- a broken word is worse than a wide line.
    """
    words = str(text or "").split()
    if not words:
        return [""]
    max_chars = max(1, int(max_chars))
    # Fewest lines that could hold it, then balance: a greedy wrap gives
    # 'Для гостей театра на' / 'Пролетарской', which reads worse than an even
    # split and paints a lopsided label.
    total = sum(len(w) for w in words) + len(words) - 1
    n_lines = max(1, -(-total // max_chars))
    while True:
        target = -(-total // n_lines)
        lines, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if cur and (len(cand) > max_chars or len(cand) > target and len(lines) < n_lines - 1):
                lines.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            lines.append(cur)
        if all(len(ln) <= max_chars for ln in lines) or n_lines >= len(words):
            return lines
        n_lines += 1


def text_geometry_lines(text: str, h_line: float = 0.0) -> tuple:
    """(w, h) for a string that may span several lines, at `h_line` per line.

    Width comes from the LONGEST line, height from the line count; the
    single-line rule (`text_geometry`) is the one-line case of this.
    """
    lines = lines_of(text)
    n = max(1, max(len(ln) for ln in lines))
    h_line = max(float(h_line or 0.0), MIN_TEXT_H)
    w = h_line * CHAR_ASPECT * n
    if w > 0.94:
        w = 0.94
        h_line = w / (CHAR_ASPECT * n)
    return w, h_line * len(lines)


MAX_TEXT_LINES = 3       # a label wraps to two or three lines; past that it is a paragraph
# The per-LINE floor for wrapped text. MIN_TEXT_H (0.08) was measured on one
# word squeezed into a box; a multi-line label stated line by line in the
# caption spelled "Для гостей / театра на / Пролетарской" correctly at a line
# height of ~0.045 in three renders on 2026-09-12 (864x1152). Holding it to
# 0.08 forced the label to 0.58 of the frame and the bottle under it could no
# longer fit in the picture whole.
MIN_LINE_H = 0.05


def fit_text_in(text: str, max_w: float, h_line: float = 0.0,
                max_lines: int = MAX_TEXT_LINES) -> tuple:
    """Wrap `text` into at most `max_lines` legible lines.

    Returns (wrapped_text, w, h). Every line is at least MIN_TEXT_H high -- the
    glyph floor is not negotiable, it is where spelling turns into scribble --
    so the width comes out as whatever the longest line needs at that size.
    The caller compares it with the room it has; if there is not enough, the
    answer is a bigger surface (a closer shot of the label), never smaller
    letters. `max_w` only caps the line length used for wrapping.
    """
    text = " ".join(str(text or "").split())
    max_w = max(0.05, float(max_w))
    floor = MIN_LINE_H if len(wrap_text(text, 10**6)) > 1 or " " in text else MIN_TEXT_H
    h_line = max(float(h_line or 0.0), floor)
    # Balanced wrap into the fewest lines whose longest line fits max_w at
    # the glyph floor; if that needs more than max_lines, use max_lines and
    # let the width overflow -- the caller grows the surface.
    n = len(text)
    per_line = max(1, int(max_w / (CHAR_ASPECT * floor)))
    lines = wrap_text(text, per_line)
    if len(lines) > max_lines:
        lines = wrap_text(text, max(per_line, -(-n // max_lines) + 2))
        while len(lines) > max_lines:
            per_line += 1
            lines = wrap_text(text, per_line)
    wrapped = "\n".join(lines)
    longest = max(len(ln) for ln in lines)
    # Use the caller's line height if it fits the width, else the floor.
    if h_line * CHAR_ASPECT * longest > max_w:
        h_line = max(floor, max_w / (CHAR_ASPECT * longest))
    w = h_line * CHAR_ASPECT * longest
    return wrapped, min(w, 0.94), h_line * len(lines)


def text_geometry(text: str, h: float = 0.0) -> tuple:
    """The (w, h) a box needs to print `text` legibly, as fractions of the frame.

    Height first (glyphs need pixels), then width from the character count. If the
    string is so long that the natural width overflows the frame, the height comes
    back DOWN to fit rather than the letters being crushed sideways — a caller that
    gets an h below MIN_TEXT_H is being told the string is too long, not offered a
    good box.
    """
    n = max(1, len(str(text or "").strip()))
    h = max(float(h or 0.0), MIN_TEXT_H)
    w = h * CHAR_ASPECT * n
    if w > 0.94:
        w = 0.94
        h = w / (CHAR_ASPECT * n)
    return w, h
