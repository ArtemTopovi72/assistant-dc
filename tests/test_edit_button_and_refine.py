"""Two live failures from 2026-07-29, both of which made the boxes unreachable.

A. The ✏️ button armed `pending_prefix = "edit the image: "` and then asked
   `_main_menu_kb` for a keyboard — the one function whose job is to CLEAR that
   prefix so the keyboard can never contradict the state. The prefix was wiped
   one line after being set. The agent received a bare complaint ("надписи
   кривые") with no instruction to edit, floundered two rounds, and told the user
   it could not call any tools. Proof in tg_activity.jsonl: the "generate" flow
   logged its prefix, the ✏️ flow logged the message without one.

B. The quality-refinement loop repainted attempt 2+ through the old model img2img.
   On an Ideogram picture that means a different model paints over a composition
   it never planned, and — worse — the delivered file carries no layout, so every
   later "move that box" silently falls back to a pixel pipeline. The image the
   user was trying to edit was old-model-inpaint_00010_.png, with no sidecar.

Run: venv/Scripts/python.exe tests/test_edit_button_and_refine.py
"""
import os, sys, re, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}\n         {detail}")


# ------------------------------------------------------ A. the ✏️ button
print("\nA. THE EDIT BUTTON KEEPS THE PREFIX IT JUST ARMED")
import tg_bot as T

# The ✏️ handler has moved twice already (tg_bot.py → tg_dispatch.py →
# tg_callbacks.py), and each move silently invalidated a hardcoded module
# list. Anchor on what the handler IS instead of where it lives: the one
# function in the tg_* family whose body arms the edit prefix. That survives
# the next move, and this suite ABORTS when nothing matches, so a stale
# anchor is loud rather than vacuous.
import ast as _ast
import glob as _glob

_ARMS = 'pending_prefix = "edit the image: "'
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_found = []
for _path in sorted(_glob.glob(os.path.join(_ROOT, "bot", "tg_*.py"))):
    _text = open(_path, encoding="utf-8").read()
    if _ARMS not in _text:
        continue
    _lines = _text.splitlines(keepends=True)
    for _node in _ast.walk(_ast.parse(_text)):
        if not isinstance(_node, _ast.FunctionDef):
            continue
        _body = "".join(_lines[_node.lineno - 1:_node.end_lineno])
        if _ARMS in _body:
            _found.append((os.path.basename(_path), _node.name, _body))
# Innermost match wins: ast.walk yields the enclosing method too when the
# handler is nested, and the shortest body is the handler itself.
_found.sort(key=lambda f: len(f[2]))
handler = _found[0][2] if _found else ""
check("the ✏️ handler is where we expect it", bool(handler.strip()),
      "no tg_*.py function arms the edit prefix — the checks below "
      "would pass vacuously")
if handler:
    print(f"       (found in {_found[0][0]}:{_found[0][1]})")
if not handler.strip():
    print("\nABORT: cannot verify the handler; refusing to report vacuous passes")
    sys.exit(1)

def _code_only(block: str) -> str:
    """Drop comment lines — the fix is explained in a comment that names the very
    function it must not call, and a naive substring check reads that as a bug."""
    return "\n".join(l for l in block.splitlines()
                     if not l.lstrip().startswith("#"))


code = _code_only(handler)
check('it arms the edit prefix', 'pending_prefix = "edit the image: "' in code,
      code[:200])
check("it does NOT ask _main_menu_kb, which clears the prefix",
      "_main_menu_kb" not in code, code[:300])
check("it uses _state_kb, which matches the armed state",
      "_state_kb" in code, code[:300])

# The invariant that made this a bug in the first place — keep it true.
mainkb = inspect.getsource(T.TelegramBot._main_menu_kb) if hasattr(T, "TelegramBot") else ""
if not mainkb:
    cls = next((v for k, v in vars(T).items()
                if isinstance(v, type) and hasattr(v, "_main_menu_kb")), None)
    mainkb = inspect.getsource(cls._main_menu_kb) if cls else ""
check("_main_menu_kb still clears pending_prefix (that is its job)",
      'pending_prefix = ""' in mainkb, mainkb[:200])

cls = next((v for k, v in vars(T).items()
            if isinstance(v, type) and hasattr(v, "_PREFIX_MENU")), None)
check("the edit prefix has a keyboard of its own to show",
      cls is not None and "edit the image: " in cls._PREFIX_MENU,
      getattr(cls, "_PREFIX_MENU", None))
check("and it is the draw keyboard",
      cls is not None and cls._PREFIX_MENU.get("edit the image: ") == "draw")

# Every OTHER place that arms a prefix must obey the same rule, or this recurs.
#
# This used to be a regex that looked 320 characters past the assignment for a
# `return`/`if`/`elif` terminator. The explanation of the fix is a 9-line
# comment sitting in exactly that gap, so the terminator was out of reach and
# the sweep matched NOTHING — a guard that silently covered zero sites, in both
# the old file and the new one. Walk the functions instead: every tg_* function
# that arms a non-empty prefix must not also ask _main_menu_kb (whose job is to
# clear it). The count is asserted so "no sites" can never again read as "all
# sites pass".
print("\n   no other site arms a prefix and then clears it")
_armers = []
for _path in sorted(_glob.glob(os.path.join(_ROOT, "bot", "tg_*.py"))):
    _text = open(_path, encoding="utf-8").read()
    if "pending_prefix" not in _text:
        continue
    _lines = _text.splitlines(keepends=True)
    for _node in _ast.walk(_ast.parse(_text)):
        if not isinstance(_node, _ast.FunctionDef):
            continue
        _body = _code_only("".join(_lines[_node.lineno - 1:_node.end_lineno]))
        _pref = re.findall(r'pending_prefix\s*=\s*"([^"]+)"', _body)
        if _pref:
            _armers.append((_node.name, _pref, _body))
# Innermost function wins; an enclosing method would drag in unrelated calls.
_armers.sort(key=lambda a: len(a[2]))
_seen = set()
for _name, _prefs, _body in _armers:
    _kept = tuple(p for p in _prefs if p.strip())
    if not _kept or _kept in _seen:
        continue
    _seen.add(_kept)
    check(f"{_name} arms {'/'.join(_kept)!r} and does not call _main_menu_kb",
          "_main_menu_kb" not in _body, _body[:180])
check("the sweep actually covered a prefix-arming site", bool(_seen),
      "no tg_* function arms a prefix — this guard is vacuous")

# ------------------------------------------- B. refinement keeps the layout
print("\nB. REFINING AN IDEOGRAM PICTURE DOES NOT REPAINT IT WITH THE OLD MODEL")
import image as I

# The refinement loop moved twice: out of generate_and_verify_image into
# generate_image_with_refinement, and out of image.py into image_generate.py.
# So locate it by its MARKER across the image module family rather than by name
# or by __module__ — an anchor that pins either one goes stale on the next move.
loop, loopname = "", ""
_mods = [I]
for _m in ("image_generate", "image_contained", "image_router"):
    try:
        _mods.append(__import__(_m))
    except Exception:
        pass
for _mod in _mods:
    for name, fn in vars(_mod).items():
        if not callable(fn) or not getattr(fn, "__module__", "").startswith("image"):
            continue
        try:
            s = inspect.getsource(fn)
        except Exception:
            continue
        if "Every attempt re-draws through Ideogram" in s:
            loop, loopname = s, name
            break
    if loop:
        break
check("the refinement loop was found", bool(loop))
check("a retry never repaints the previous picture (that lost the layout)",
      "previous_image_path" not in loop and "use_inpaint" not in loop,
      "the refinement loop can still hand a previous image to the renderer")

# A picture with no layout must still be diagnosable rather than mysterious.
draw = inspect.getsource(I._ideogram_draw)
check("drawing without a recorded layout is logged loudly",
      "WITHOUT a recorded layout" in draw, draw[-400:])
check("the layout is recorded regardless of the lettering-rounds setting",
      "rounds and any(" in draw and "if True:" not in draw, draw[:400])

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
