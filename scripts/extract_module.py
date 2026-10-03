"""AST-driven extraction of top-level definitions from one module into another.

Usage:
    python scripts/extract_module.py SRC DST NAME [NAME...]

Moves the given top-level names (functions, classes, module-level assignments,
including annotated assignments) out of SRC and into DST, preserving order and
any leading comment block attached to each definition.  Prints the set of free
names the moved code references so the import header can be computed from what
the code actually uses rather than by eye.
"""
import ast
import sys
from pathlib import Path


def node_name(n):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [n.name]
    if isinstance(n, ast.Assign):
        out = []
        for t in n.targets:
            if isinstance(t, ast.Name):
                out.append(t.id)
            elif isinstance(t, ast.Tuple):
                out += [e.id for e in t.elts if isinstance(e, ast.Name)]
        return out
    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
        return [n.target.id]
    return []


def segments(src_path, names):
    text = Path(src_path).read_text(encoding="utf-8")
    lines = text.split("\n")
    tree = ast.parse(text)
    body = tree.body
    picked = []
    for i, n in enumerate(body):
        if not (set(node_name(n)) & set(names)):
            continue
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        # absorb the leading comment/blank block, but never cross into the
        # previous top-level statement
        floor = body[i - 1].end_lineno + 1 if i else 1
        s = start
        while s - 1 > floor:
            prev = lines[s - 2].strip()
            if prev.startswith("#") or prev == "":
                s -= 1
            else:
                break
        while s < start and lines[s - 1].strip() == "":
            s += 1
        picked.append((s, n.end_lineno, n))
    return text, lines, picked


def free_names(nodes):
    """Names referenced by the moved code that it does not itself bind."""
    bound, used = set(), set()
    for n in nodes:
        bound |= set(node_name(n))
        for sub in ast.walk(n):
            if isinstance(sub, ast.Name):
                (bound if isinstance(sub.ctx, ast.Store) else used).add(sub.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(sub.name)
            elif isinstance(sub, ast.arg):
                bound.add(sub.arg)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for a in sub.names:
                    bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(sub, ast.Attribute):
                pass
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                bound.add(sub.name)
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                bound |= set(sub.names)
    return sorted(used - bound - set(dir(__builtins__)) - set(dir(int)))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    _, lines, picked = segments(src, names)
    found = set()
    for _, _, n in picked:
        found |= set(node_name(n))
    missing = set(names) - found
    if missing:
        sys.exit(f"NOT FOUND in {src}: {sorted(missing)}")

    chunks = ["\n".join(lines[s - 1:e]) for s, e, _ in picked]
    Path(dst + ".body").write_text("\n\n\n".join(chunks) + "\n", encoding="utf-8")

    drop = set()
    for s, e, _ in picked:
        drop |= set(range(s, e + 1))
    kept = [l for i, l in enumerate(lines, 1) if i not in drop]
    Path(src + ".stripped").write_text("\n".join(kept), encoding="utf-8")

    import builtins
    fn = [n for n in free_names([n for _, _, n in picked])
          if not hasattr(builtins, n)]
    print("MOVED:", " ".join(sorted(found)))
    print("FREE NAMES:", " ".join(fn))


if __name__ == "__main__":
    main()
