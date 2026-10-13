"""Suites hand their fakes in (on the context, the bot, a parameter) instead of writing
over a module that every later suite in the process imports too: a stub left on
`video.generate_video` by one suite failed another three files later (2026-10-13).

A ratchet: the count of module patches in tests/ may only go down. Lower BASELINE as
suites move to seams; a new patch fails here with its file and line.

Run: venv/Scripts/python.exe -m pytest -q tests/test_no_module_patching.py
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = 1081

_IMPORT = re.compile(r"^\s*import ([\w.]+)(?: as (\w+))?|^\s*from [\w.]+ import (\w+)(?: as (\w+))?", re.M)


# The other ways a suite writes over what every later suite imports: a module swapped in
# sys.modules, setattr( on an imported module, the rebinding helpers.
_OTHER = re.compile(r"sys\.modules\[[^]]+\] = |monkeypatch\.setitem\(sys\.modules"
                    r"|\b(?:RebindPatches|_dr_patch|fake_modules|setattr_saved)\(")


def module_patches() -> list:
    """(file:line, text) of every monkeypatch.setattr / setenv, every `mod.attr = ...` on a
    name the file imported, and every _OTHER write."""
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "tests", "*.py"))):
        if os.path.basename(path) == os.path.basename(__file__):
            continue
        src = open(path, encoding="utf-8").read()
        names = set()
        for m in _IMPORT.finditer(src):
            names.add(m.group(2) or (m.group(1) or "").split(".")[0] or m.group(4) or m.group(3))
        names.discard(None)
        assign = re.compile(r"^\s*(%s)\.\w+(?:\.\w+)* = " % "|".join(map(re.escape, names))) if names else None
        # setattr on an imported module (a loop over a test's own objects is not one)
        bare = re.compile(r"^\s*setattr\((%s)[,.]" % "|".join(map(re.escape, names))) if names else None
        for i, line in enumerate(src.splitlines(), 1):
            if ("monkeypatch.setattr(" in line or "monkeypatch.setenv(" in line
                    or (assign and assign.match(line)) or (bare and bare.match(line))
                    or _OTHER.search(line)):
                out.append((f"{os.path.basename(path)}:{i}", line.strip()))
    return out


def test_module_patches_only_go_down():
    found = module_patches()
    assert len(found) <= BASELINE, (f"{len(found)} module patches, the ratchet is at {BASELINE}: hand "
                                    f"the fake in instead. Newest suspects: {found[-5:]}")
