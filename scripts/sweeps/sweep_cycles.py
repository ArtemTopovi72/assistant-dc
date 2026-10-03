"""Import-cycle sweep over the project's own top-level modules.

Distinguishes MODULE-LEVEL imports (the ones that can actually deadlock/ImportError
at import time) from FUNCTION-LEVEL ones (a deliberate cycle break). Only
module-level edges form the graph whose cycles matter.
"""
import ast
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

local = set()
for f in os.listdir(ROOT):
    if f.endswith(".py") and not f.startswith("_"):
        local.add(f[:-3])


def edges_of(path, mod):
    """Return (toplevel_edges, deferred_edges) as {target: [lineno,...]}."""
    top, deferred = defaultdict(list), defaultdict(list)
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return top, deferred
    # Mark every node that sits inside a function/lambda body.
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                inside.add(id(sub))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
        for n in names:
            if n in local and n != mod:
                (deferred if id(node) in inside else top)[n].append(node.lineno)
    return top, deferred


TOP, DEF = {}, {}
for mod in sorted(local):
    TOP[mod], DEF[mod] = edges_of(os.path.join(ROOT, mod + ".py"), mod)

# Tarjan SCC over module-level edges only.
index, low, on, stack, order, sccs = {}, {}, set(), [], [0], []


def strongconnect(v):
    index[v] = low[v] = order[0]
    order[0] += 1
    stack.append(v)
    on.add(v)
    for w in TOP.get(v, {}):
        if w not in index:
            strongconnect(w)
            low[v] = min(low[v], low[w])
        elif w in on:
            low[v] = min(low[v], index[w])
    if low[v] == index[v]:
        comp = []
        while True:
            w = stack.pop()
            on.discard(w)
            comp.append(w)
            if w == v:
                break
        sccs.append(comp)


sys.setrecursionlimit(10000)
for m in sorted(TOP):
    if m not in index:
        strongconnect(m)

print("=" * 72)
print("MODULE-LEVEL IMPORT CYCLES (strongly-connected components, size > 1)")
print("=" * 72)
found = 0
for comp in sccs:
    if len(comp) > 1:
        found += 1
        print("\nCYCLE of %d: %s" % (len(comp), ", ".join(sorted(comp))))
        cs = set(comp)
        for a in sorted(comp):
            for b in sorted(TOP[a]):
                if b in cs:
                    print("    %-26s -> %-26s  (line %s)"
                          % (a, b, ", ".join(map(str, TOP[a][b]))))
if not found:
    print("\nNone.")

print("\n" + "=" * 72)
print("SELF-REFERENTIAL / DEFERRED EDGES that exist ONLY inside functions")
print("(these are cycle BREAKS already in place — a module-level import here")
print(" would re-create a cycle)")
print("=" * 72)
for a in sorted(DEF):
    for b in sorted(DEF[a]):
        # a defers b; does b import a at module level? then this defer is load-bearing
        if a in TOP.get(b, {}):
            print("  %-26s defers %-26s (line %s)  <- b imports a at module level (line %s)"
                  % (a, b, DEF[a][b][0], TOP[b][a][0]))
