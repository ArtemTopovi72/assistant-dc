"""Property fuzz over the Telegram text pipeline: markdown -> HTML -> chunks.

Why a fuzzer and not more examples: Telegram does not degrade malformed markup,
it REJECTS the message. A single crossed or half-written tag anywhere in a long
research report costs the user the entire answer, and the delivery path logs the
failure without retrying. So the properties below are correctness-critical, and
the failures that motivated this file were all found by random search, not by
reading the regexes:

  * "# ```\n```"            -> <b><pre></b>\n</pre>          (heading crossed a fenced block)
  * "[`](http://)`"         -> <a ...><code></a></code>      (link rule split a code span)
  * "<http://`e`>"          -> href="http://<code>e</code>"  (markup inside an attribute)
  * "[<http://x>**](http://)**" -> nested <a> inside <a>
  * an oversized line cut at a space that was INSIDE a tag

Properties, for ANY input:
  P1 every chunk fits the Telegram limit
  P2 every chunk is well-formed: no half-written tag, no stray close, no crossing
  P3 the visible text survives chunking (nothing silently dropped)
  P4 no exception

Run: venv/Scripts/python.exe tests/test_tg_text_fuzz.py
"""
import os
import random
import re
import sys
import html as _html

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tempfile
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_textfuzz_"))

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1; print("[PASS] " + name)
    else:
        BAD += 1; print("[FAIL] " + name + (" - " + detail if detail else ""))


LIMIT = 4096

# Fragments chosen to collide: every markup dialect the models emit, plus the
# characters that escaping/placeholders are built out of.
ALPHA = [
    "hello", "мир", "текст", "word", "日本語", "a", "  ", "\n", "\n\n", "\t",
    "**bold**", "*it*", "_us_", "__b__", "`code`", "# Head", "## Sub", "- item",
    "```\ncode block\n```", "```py\nx = 1\n```",
    "[l](https://e.example/a_b(c))", "[`x`](https://e.example/q?a=1&b=2)",
    "<https://e.example/x>", "https://e.example/" + "z" * 150,
    "&", "<", ">", '"', "'", "\\", "\x00", "…", "—", "]", "[", "(", ")",
    "<b>", "</i>", "<a href=\"x\">", "*", "_", "`", "#", "|", "~",
    # Bare, unpaired delimiters: a crossing needs the dialects to INTERLEAVE
    # ("_**x_**"), which self-contained fragments like "**bold**" never produce.
    "**", "__", "~~", "abc", "x",
]


def visible(s):
    return re.sub(r"\s+", "", _html.unescape(re.sub(r"<[^>]*>", "", s)))


def well_formed(chunk):
    """Telegram's actual requirement: complete, properly nested tags."""
    if chunk.count("<") != chunk.count(">"):
        return False, "half-written tag"
    stack = []
    for m in re.finditer(r"<(/?)(b|i|u|s|code|pre|a)(\s[^>]*)?>", chunk, re.I):
        if m.group(1):
            if not stack or stack[-1] != m.group(2).lower():
                return False, "stray/crossed close </%s>" % m.group(2)
            stack.pop()
        else:
            stack.append(m.group(2).lower())
    if stack:
        return False, "unclosed %s" % stack
    return True, ""


def run_fuzz(seed, iterations, max_fragments):
    rnd = random.Random(seed)
    worst = {}
    for _ in range(iterations):
        src = "".join(rnd.choice(ALPHA)
                      for _ in range(rnd.randint(1, max_fragments)))
        try:
            h = T._md_to_html(src)
            chunks = T._split_html(h, limit=LIMIT)
        except Exception as exc:
            worst.setdefault("P4", (f"{type(exc).__name__}: {exc}", src))
            continue
        for c in chunks:
            if len(c) > LIMIT:
                worst.setdefault("P1", (f"chunk of {len(c)} chars", src))
            ok, why = well_formed(c)
            if not ok:
                worst.setdefault("P2", (why + " in " + repr(c[-90:]), src))
        if visible("".join(chunks)) != visible(h):
            worst.setdefault("P3", ("visible text changed", src))
    return worst


NAMES = {"P1": "every chunk fits the Telegram limit",
         "P2": "every chunk is well-formed markup",
         "P3": "chunking preserves the visible text",
         "P4": "no input raises"}

for _seed, _iters, _frags in ((20260728, 6000, 400), (1, 6000, 60), (99991, 3000, 900)):
    _worst = run_fuzz(_seed, _iters, _frags)
    for _p in ("P1", "P2", "P3", "P4"):
        _hit = _worst.get(_p)
        check(f"seed {_seed}: {NAMES[_p]}", _hit is None,
              "" if _hit is None else f"{_hit[0]} | source={_hit[1][:220]!r}")

# The specific historical failures, pinned as examples so a regression names
# itself instead of surfacing as an opaque random seed.
REGRESSIONS = [
    ("# ```\n```",                      "heading crossing a fenced block"),
    ("[`](http://)`",                   "link rule splitting a code span"),
    ("<http://`e`>",                    "markup restored inside an href"),
    ("[<http://x>**](http://)**",       "autolink nested inside a link"),
    ("a \x000\x00 b _x_ c",             "NUL colliding with the stash marker"),
    ("_**<http://x>_**",                "underscore and asterisk spans crossing"),
    # …and the same crossing without a URL in the body: the italic rule excludes
    # "/", so a link in the middle masks the bug rather than triggering it.
    ("_**abc_**",                       "underscore span crossing a bold span"),
    ("*__x*__",                         "asterisk span crossing an underscore span"),
]
for _src, _why in REGRESSIONS:
    try:
        _chunks = T._split_html(T._md_to_html(_src), limit=LIMIT)
        _bad = [well_formed(c)[1] for c in _chunks if not well_formed(c)[0]]
        check("regression: " + _why, not _bad, "; ".join(_bad))
    except Exception as exc:
        check("regression: " + _why, False, f"{type(exc).__name__}: {exc}")

# This one must bypass _md_to_html: the point is a real tag reaching _split_html,
# and running the source through the converter would escape it into plain text —
# which is how an earlier version of this check passed against the bug.
_RAW_SPLIT = [
    # The tag must sit so that the cut point falls INSIDE its attribute: shifting
    # the URL length moves the boundary and the bug hides.
    ("x" * 4000 + ' <a href="https://e.com/a b c">tail</a>' + "y" * 500,
     "hard split at a space inside a tag"),
    ("x" * 4000 + ' <a href="https://e.com/a b c d e f">tail</a>' + "y" * 500,
     "hard split at a space inside a longer tag"),
    ("<b>" + "word " * 900 + '<a href="https://e.example/x?a=1 2">link</a>'
     + "tail " * 200 + "</b>",
     "hard split near a spaced attribute mid-document"),
]
for _html_src, _why in _RAW_SPLIT:
    _chunks = T._split_html(_html_src, limit=LIMIT)
    _bad = [well_formed(c)[1] for c in _chunks if not well_formed(c)[0]]
    check("regression: " + _why, not _bad, "; ".join(_bad))
    check("regression: " + _why + " (within limit)",
          all(len(c) <= LIMIT for c in _chunks),
          str(max(len(c) for c in _chunks)))

# The reopen prefix is every open tag carried across a boundary. A fixed 64-byte
# reserve was not enough for one long tracking URL nested in b/i/u/s, and the
# caller truncates at the limit — cutting the closing tags off, which Telegram
# rejects. Sweep label lengths so the near-boundary cases are actually hit.
_URL = "https://example.com/very/long/tracking/url?" + "&".join(
    f"utm_param_{i}=value_{i}" for i in range(14))
_worst_over = None
for _n in range(600, 3000, 7):
    _label = " ".join(f"word{i}" for i in range(_n))
    _chunks = T._split_html(
        f'<b><i><u><s><a href="{_URL}">{_label}</a></s></u></i></b>', limit=LIMIT)
    _big = [len(c) for c in _chunks if len(c) > LIMIT]
    if _big:
        _worst_over = (_n, max(_big))
        break
check("a carried-over long href never pushes a chunk past the limit",
      _worst_over is None,
      "" if _worst_over is None
      else f"{_worst_over[1]} chars at label width {_worst_over[0]}")

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
