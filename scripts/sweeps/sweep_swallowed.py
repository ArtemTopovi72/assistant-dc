"""Swallowed-exception sweep: find handlers that can hide a defect.

Motivated by a real one. `utils.throttle_external_calls` raised AttributeError on
a context with no `api_lock`; `llm.py` caught it, logged it and returned None, so
every model call on that path silently became "no answer" while the calling suite
still exited 0. The handler was correct — the sweep exists to find the handlers
where it is NOT.

Three severities, worst first:

  SILENT   `except ...: pass` (or `return`/`continue`/`None`) with no logging and
           no comment. Nothing anywhere records that something failed.
  BARE     `except:` / `except BaseException:` — also catches KeyboardInterrupt
           and SystemExit, so Ctrl-C inside the block is swallowed too.
  BROAD    `except Exception` around a body of 12+ statements. The wider the try,
           the less likely the handler is right for everything it covers.

A handler that logs, re-raises, or carries a `# pragma: no cover` / explanatory
comment on the `except` line is not reported: someone thought about it.

Run: venv/Scripts/python.exe scripts/sweeps/sweep_swallowed.py [--all]
Without --all, only Lane 5 files are printed in full; the rest are counted.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANE = {"config", "llm", "graph", "graph_personality", "graph_fastpath", "graph_finalize",
        "tools", "utils", "audio", "remix",
        "mashup_stems", "music", "comfy_client"}
SKIP_DIRS = {"venv", "models", "runtime", "outputs", ".git", "tg_images",
             "tg_libraries", "tg_memory", "tg_users_backups", "__pycache__"}
BROAD_BODY = 12


def _logs(node):
    """True if the handler body records the failure somehow."""
    for n in ast.walk(node):
        if isinstance(n, ast.Raise):
            return True
        if isinstance(n, ast.Call):
            f = n.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in ("debug", "info", "warning", "warn", "error", "exception",
                        "critical", "print", "log", "_log"):
                return True
    return False


def _annotated(lines, handler):
    """A comment on or just above the `except` line counts as a decision."""
    i = handler.lineno - 1
    if "#" in lines[i]:
        return True
    j = i - 1
    return j >= 0 and lines[j].strip().startswith("#")


def scan(path, rel):
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception:
        return []
    lines = src.splitlines()
    out = []
    for t in ast.walk(tree):
        if not isinstance(t, ast.Try):
            continue
        n_stmts = sum(1 for _ in ast.walk(ast.Module(body=t.body, type_ignores=[]))
                      if isinstance(_, ast.stmt))
        for h in t.handlers:
            bare = h.type is None or (isinstance(h.type, ast.Name)
                                      and h.type.id == "BaseException")
            body_is_noop = all(
                isinstance(s, ast.Pass)
                or (isinstance(s, ast.Return) and (s.value is None
                    or (isinstance(s.value, ast.Constant) and s.value.value is None)))
                or isinstance(s, (ast.Continue, ast.Break))
                for s in h.body)
            silent = body_is_noop and not _logs(h) and not _annotated(lines, h)
            broad = (isinstance(h.type, ast.Name) and h.type.id == "Exception"
                     and n_stmts >= BROAD_BODY and not _logs(h))
            if silent:
                sev = "SILENT"
            elif bare:
                sev = "BARE  "
            elif broad:
                sev = "BROAD "
            else:
                continue
            out.append((sev, rel, h.lineno, n_stmts))
    return out


def main():
    show_all = "--all" in sys.argv
    findings = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT).replace("\\", "/")
            if rel.startswith("tests/") or rel.startswith("scripts/"):
                continue
            findings += scan(os.path.join(dirpath, fn), rel)

    lane = [f for f in findings if f[1].rsplit("/", 1)[-1][:-3] in LANE]
    rest = [f for f in findings if f not in lane]
    print("=" * 74)
    print("SWALLOWED-EXCEPTION SWEEP")
    print("=" * 74)
    for sev in ("SILENT", "BARE  ", "BROAD "):
        print(f"\n--- {sev.strip()} : {sum(1 for f in findings if f[0] == sev)} total "
              f"({sum(1 for f in lane if f[0] == sev)} in lane)")
        rows = sorted(f for f in (findings if show_all else lane) if f[0] == sev)
        for _s, rel, ln, n in rows:
            print(f"    {rel}:{ln}   try-body {n} stmt(s)")
    print(f"\n{len(findings)} handler(s) flagged across the tree; {len(lane)} in Lane 5 files.")
    if not show_all:
        print("Pass --all to list the other %d." % len(rest))


if __name__ == "__main__":
    main()
