r"""Write accepted SkillOpt edits into tool_descriptions.py -- the reviewed step.

    venv/Scripts/python bench/skillopt_apply.py [runtime/skillopt/best_edits.json] [--dry]

bench/skillopt.py never touches the source; it leaves the edits that raised the
held-out score. This finds, for each edited tool, the *_DESC constant whose
value IS that tool's description, recomputes the new text exactly the way the
optimizer did (bench.skillopt.apply_edits), and rewrites only that constant.
--dry prints the before/after instead of writing.
"""
import ast
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _constants(src):
    """{NAME: (value, start_line, end_line)} for every top-level string constant."""
    out = {}
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                val = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(val, str):
                out[node.targets[0].id] = (val, node.lineno, node.end_lineno)
    return out


def render(name, text, width=84):
    lines = textwrap.wrap(text, width=width, drop_whitespace=False) or [""]
    body = "\n".join("    " + json.dumps(l, ensure_ascii=False) for l in lines)
    return f"{name} = (\n{body}\n)"


def apply_to_source(src, schemas, edits):
    from bench.skillopt import apply_edits
    new_schemas = apply_edits(schemas, edits)
    old = {s["function"]["name"]: s["function"].get("description", "") for s in schemas}
    new = {s["function"]["name"]: s["function"].get("description", "") for s in new_schemas}
    consts = _constants(src)
    lines = src.splitlines()
    changes = []
    for tool in sorted({e["tool"] for e in edits}):
        hits = [(n, a, b) for n, (v, a, b) in consts.items() if v == old[tool]]
        if len(hits) != 1:
            raise ValueError(f"{tool}: {len(hits)} constants hold its description")
        changes.append((hits[0], new[tool]))
    for (name, a, b), text in sorted(changes, key=lambda c: -c[0][1]):
        lines[a - 1:b] = render(name, text).splitlines()
    return "\n".join(lines) + "\n", changes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    edits = json.loads(Path(args[0] if args else ROOT / "runtime/skillopt/best_edits.json")
                       .read_text(encoding="utf-8"))
    import tools
    path = ROOT / "agent/tool_descriptions.py"
    src, changes = apply_to_source(path.read_text(encoding="utf-8"), tools.TOOL_SCHEMAS, edits)
    for (name, _, _), text in changes:
        print(f"== {name}\n{text}\n")
    if "--dry" not in sys.argv:
        path.write_text(src, encoding="utf-8")
        print(f"wrote {len(changes)} description(s) to {path.name}")


if __name__ == "__main__":
    main()
