"""Duplication sweep: functions with structurally identical bodies.

Normalizes each function to its AST shape with all identifiers, constants and
docstrings erased, so two functions that differ only in naming still collide.
Reports groups of 2+ functions sharing a shape, ordered by size (biggest
duplicated body first — that is where sharing pays).
"""
import ast
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP_DIRS = {"venv", "models", "runtime", "outputs", "tg_images", "tg_libraries",
             "tg_memory", "tg_users_backups", ".git", "__pycache__", "assets",
             "personalities", "docs", "scripts", "tv-adb", ".claude", "tests"}
MIN_NODES = 18   # ignore trivial one-liners / property stubs

LANE = {"config", "llm", "graph", "graph_personality", "tools", "utils",
        "audio", "mashup", "music", "comfy_client"}


def shape(node):
    """AST shape string: structure only, names and literals erased."""
    out = []

    def walk(n, depth):
        if isinstance(n, ast.expr_context):
            return
        out.append(type(n).__name__)
        for child in ast.iter_child_nodes(n):
            walk(child, depth + 1)
    for stmt in node.body:
        # drop a leading docstring
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            continue
        walk(stmt, 0)
    return "|".join(out)


groups = defaultdict(list)
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if not fn.endswith(".py") or fn.endswith((".bak", ".bak9")):
            continue
        path = os.path.join(dirpath, fn)
        mod = os.path.relpath(path, ROOT).replace("\\", "/")
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s = shape(node)
                n_nodes = s.count("|") + 1
                if n_nodes >= MIN_NODES:
                    groups[s].append((mod, node.name, node.lineno, n_nodes))

dupes = [(v[0][3], v) for v in groups.values() if len(v) > 1]
dupes.sort(reverse=True)

print("=" * 74)
print("STRUCTURALLY IDENTICAL FUNCTION BODIES (>= %d AST nodes)" % MIN_NODES)
print("  [LANE] = at least one copy is in a Lane 5 file")
print("=" * 74)
shown = 0
for size, v in dupes:
    mods = {m.split("/")[-1][:-3] for m, _, _, _ in v}
    tag = "LANE" if mods & LANE else "--"
    print("\n[%s] %d copies, %d AST nodes each:" % (tag, len(v), size))
    for mod, name, lineno, _ in sorted(v):
        print("        %-34s %-34s line %d" % (mod, name, lineno))
    shown += 1
    if shown >= 30:
        print("\n  ... (truncated)")
        break
print("\n  %d duplicated shape(s) found." % len(dupes))
