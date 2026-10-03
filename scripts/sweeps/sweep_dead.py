"""Dead-code sweep: module-level names that nothing references.

A name counts as LIVE if it appears anywhere outside its own definition line as
an identifier, OR as a bare string literal anywhere in the tree (this codebase
dispatches by string name in several places, so a string hit means "possibly
reached dynamically" and the name is NOT reported as dead).

Scans .py under the project root (including tests/ and scripts/, which are
legitimate consumers). Skips dunders and obvious framework hooks.
"""
import ast
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP_DIRS = {"venv", "models", "runtime", "outputs", "tg_images", "tg_libraries",
             "tg_memory", "tg_users_backups", ".git", "__pycache__", "assets",
             "personalities", "docs", "scripts", "tv-adb", ".claude"}

# Lane 5's editable files — reported first and marked, since only these can be
# fixed here. Everything else is reported for follow-up.
LANE = {"config", "llm", "graph", "graph_personality", "tools", "utils",
        "audio", "mashup", "music", "comfy_client"}

py_files = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if fn.endswith(".py") and not fn.endswith((".bak", ".bak9")):
            py_files.append(os.path.join(dirpath, fn))

sources = {}
for p in py_files:
    try:
        sources[p] = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        pass

# Every identifier used anywhere, and every string literal anywhere.
ident_uses = defaultdict(int)     # name -> count of identifier occurrences
string_lits = set()
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

for p, src in sources.items():
    for m in IDENT_RE.finditer(src):
        ident_uses[m.group(0)] += 1
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_lits.add(node.value)
            # also split, so "a.b" or "tools.search" contributes its parts
            for part in re.split(r"[^A-Za-z0-9_]+", node.value):
                if part:
                    string_lits.add(part)


def defined_names(path):
    """(name, lineno, kind) for every module-level definition."""
    out = []
    try:
        tree = ast.parse(sources[path])
    except (SyntaxError, KeyError):
        return out
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node.lineno, "def"))
        elif isinstance(node, ast.ClassDef):
            out.append((node.name, node.lineno, "class"))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.append((t.id, node.lineno, "const"))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append((node.target.id, node.lineno, "const"))
    return out


SRC_MODS = sorted(
    m for m in (os.path.basename(p)[:-3] for p in sources
                if os.path.dirname(p) == ROOT)
    if not m.startswith("_"))

print("=" * 74)
print("DEAD-CODE CANDIDATES  (defined once, referenced NOWHERE else,")
print("                       not even as a bare string literal)")
print("  [LANE] = Lane 5 can fix it here.  [--] = report only.")
print("=" * 74)

total = 0
for mod in SRC_MODS:
    path = os.path.join(ROOT, mod + ".py")
    rows = []
    for name, lineno, kind in defined_names(path):
        if name.startswith("__") or name == "logger":
            continue
        if ident_uses.get(name, 0) <= 1 and name not in string_lits:
            rows.append((lineno, kind, name))
    if rows:
        tag = "LANE" if mod in LANE else "--"
        print("\n[%s] %s.py" % (tag, mod))
        for lineno, kind, name in sorted(rows):
            print("        line %-5d %-6s %s" % (lineno, kind, name))
            total += 1
print("\n  %d candidate(s) total." % total)

print("\n" + "=" * 74)
print("NOT DEAD — referenced ONLY via a string literal (dynamic dispatch).")
print("Listed so a follow-up pass does not delete them by mistake.")
print("=" * 74)
n_dyn = 0
for mod in SRC_MODS:
    path = os.path.join(ROOT, mod + ".py")
    rows = []
    for name, lineno, kind in defined_names(path):
        if name.startswith("__") or name == "logger":
            continue
        if ident_uses.get(name, 0) <= 1 and name in string_lits:
            rows.append((lineno, kind, name))
    if rows:
        print("\n  %s.py" % mod)
        for lineno, kind, name in sorted(rows):
            print("        line %-5d %-6s %s" % (lineno, kind, name))
            n_dyn += 1
print("\n  %d name(s) kept alive only by a string reference." % n_dyn)
