"""Architecture invariants for the top-level module layer.

The tg_* package is a deliberate import cycle: tg_bot pulls in twelve mixins
with `from tg_x import XMixin`, and each mixin ends with a bare `import tg_bot`
so it can reach module-level helpers at call time.  That works in BOTH import
orders only because of two properties nothing was checking:

  * the back-edge is a bare `import tg_bot`, never `from tg_bot import name`
    -- a from-import needs the module fully executed, and in a cycle it is not;
  * it sits BELOW every class in the mixin, so when tg_bot is imported first
    the mixin class already exists by the time tg_bot asks for it.

Break either one and the package still imports in the order tests happen to
use, then dies at startup in the other order.  These checks are static and
cheap; the subprocess check at the end proves the static rule matches reality.

Run: venv/Scripts/python.exe tests/test_module_architecture.py
"""
import ast, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

# Symbols that are intentionally defined without an in-repo caller.  Add a name
# here only with the reason it exists -- an empty allowlist is the default, and
# a growing one means dead code is accumulating again.
UNREFERENCED_ALLOWED: dict = {}

# ".claude" matters as much as "venv" here: it holds agent worktrees, which are
# whole stale copies of this repo.  Walking them makes every symbol look
# referenced by its own old duplicate, so the dead-symbol check silently stops
# finding anything.
_SKIP_DIRS = {"venv", "models", "models_ext", "runtime", "outputs", "tg_images", "tg_libraries",
              "tg_memory", "tg_users_backups", ".git", ".claude", "__pycache__",
              "dist", "out", "_gputest", "vocos", "tv-adb", "speakers", "assets",
              "memory", "personalities", "tg_mashup", "docs", "scripts", "tools"}


def _top_modules():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from install_paths import source_dirs
    return {f[:-3]: os.path.join(d, f) for d in source_dirs() for f in os.listdir(d)
            if f.endswith(".py") and not f.startswith("_")}


def _parse(path):
    return ast.parse(open(path, encoding="utf-8").read())


def _top_level_edges(mods):
    """module -> {imported module: (lineno, is_bare_import)} for module-scope imports."""
    edges = {}
    for m, p in mods.items():
        inside = set()
        tree = _parse(p)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    inside.add(id(sub))
        out = {}
        for node in ast.walk(tree):
            if id(node) in inside:
                continue          # a lazy import inside a function cannot cycle
            if isinstance(node, ast.Import):
                for a in node.names:
                    n = a.name.split(".")[0]
                    if n in mods and n != m:
                        out[n] = (node.lineno, True)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                n = node.module.split(".")[0]
                if n in mods and n != m:
                    out.setdefault(n, (node.lineno, False))
        edges[m] = out
    return edges


def _cycles(edges):
    found, seen = [], set()
    def dfs(node, path):
        for nxt in sorted(edges.get(node, {})):
            if nxt in path:
                cyc = path[path.index(nxt):]
                key = tuple(sorted(set(cyc)))
                if key not in seen:
                    seen.add(key); found.append(cyc)
            elif len(path) < 8:
                dfs(nxt, path + [nxt])
    for m in sorted(edges):
        dfs(m, [m])
    return found


def test_every_cycle_has_a_bare_back_edge():
    """A cycle made only of `from X import name` edges cannot be imported both ways."""
    mods = _top_modules()
    edges = _top_level_edges(mods)
    cycles = _cycles(edges)
    check("cycles_found_at_all", True, str(len(cycles)) + " top-level cycles")
    bad = []
    for cyc in cycles:
        pairs = [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]
        if not any(edges[a][b][1] for a, b in pairs):
            bad.append(" -> ".join(cyc + [cyc[0]]))
    check("every_cycle_has_a_bare_import_edge", not bad,
          "from-import-only cycles: " + "; ".join(bad))


def test_cycle_back_edges_sit_below_the_definitions():
    """The bare `import X` that closes a cycle must come after the last class/def.

    Otherwise importing the mixin first re-enters tg_bot before the mixin class
    exists, and tg_bot's `from tg_x import XMixin` raises ImportError.
    """
    mods = _top_modules()
    edges = _top_level_edges(mods)
    in_cycle = set()
    for cyc in _cycles(edges):
        in_cycle.update(cyc)
    offenders, checked = [], 0
    for m in sorted(in_cycle):
        tree = _parse(mods[m])
        defs = [n.lineno for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        last_def = max(defs) if defs else 0
        for target, (lineno, bare) in edges[m].items():
            if not bare or target not in in_cycle:
                continue
            checked += 1
            if lineno < last_def:
                offenders.append(m + ":" + str(lineno) + " import " + target
                                 + " (last def at " + str(last_def) + ")")
    check("back_edges_checked", checked >= 12, "only " + str(checked) + " bare cycle imports seen")
    check("back_edges_below_definitions", not offenders, "; ".join(offenders))


def test_no_from_import_across_a_back_edge():
    """No module may reach into tg_bot with `from tg_bot import ...` at module scope."""
    mods = _top_modules()
    edges = _top_level_edges(mods)
    # Only module scope counts.  gui_telegram_tab does `from tg_bot import
    # TelegramBot` inside _start() on purpose -- a lazy import runs long after
    # tg_bot finished executing, so it can never see a half-built module.
    bad = [m + ":" + str(e["tg_bot"][0])
           for m, e in edges.items()
           if m != "tg_bot" and "tg_bot" in e and not e["tg_bot"][1]]
    check("no_module_scope_from_tg_bot", not bad, ", ".join(bad))


def test_cycle_members_import_standalone():
    """Import each cycle member first, in a fresh interpreter -- the real proof."""
    mods = _top_modules()
    edges = _top_level_edges(mods)
    members = sorted({m for cyc in _cycles(edges) for m in cyc})
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", F5_TEST_RUN="1")
    failures = []
    for m in members:
        r = subprocess.run([sys.executable, "-c", "import " + m], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()
            failures.append(m + ": " + (tail[-1] if tail else "no stderr"))
    check("cycle_members_checked", len(members) >= 12, str(len(members)) + " members")
    check("every_cycle_member_imports_first", not failures, "; ".join(failures))


def test_no_unreferenced_top_level_symbols():
    """A top-level def/class nobody names anywhere -- including tests -- is dead."""
    files = []
    for dp, dns, fns in os.walk(ROOT):
        # any virtualenv (venv_diar, venv_mula, ...), not just the one named "venv"
        dns[:] = [d for d in dns if d not in _SKIP_DIRS
                  and not os.path.exists(os.path.join(dp, d, "pyvenv.cfg"))]
        files += [os.path.join(dp, f) for f in fns if f.endswith(".py")]
    seen = {}
    for f in files:
        for tok in re.findall(r"[A-Za-z_][A-Za-z_0-9]*",
                              open(f, encoding="utf-8", errors="replace").read()):
            seen[tok] = seen.get(tok, 0) + 1
    dead = []
    for f in files:
        rel = os.path.relpath(f, ROOT)
        if rel.startswith("tests" + os.sep) or rel.startswith("bench" + os.sep):
            continue
        try:
            tree = _parse(f)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                n = node.name
                if n.startswith("__") or n in UNREFERENCED_ALLOWED:
                    continue
                if seen.get(n, 0) <= 1:
                    dead.append(rel + ":" + str(node.lineno) + " " + n)
    check("dead_scan_saw_the_repo", len(files) > 100, str(len(files)) + " files scanned")
    check("no_unreferenced_top_level_symbols", not dead, "; ".join(dead))


def test_pth_puts_source_folders_before_site_packages():
    """An installed package named like one of our modules (pip's `lmstudio`) must not win."""
    import install_paths, tempfile
    site_dir = tempfile.mkdtemp()
    real = install_paths.site.getsitepackages
    install_paths.site.getsitepackages = lambda: [site_dir]
    try:
        line = open(install_paths.ensure(), encoding="utf-8").read()
    finally:
        install_paths.site.getsitepackages = real
    import site as _site
    saved = sys.path[:]
    try:
        sys.path[:] = [site_dir]
        _site.addpackage(site_dir, "assistant_dc.pth", set())
        check("pth_prepends_source_folders", sys.path[-1] == site_dir and len(sys.path) > 1, repr(line[:80]))
    finally:
        sys.path[:] = saved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception:
            failed += 1
            import traceback; traceback.print_exc()
    passed = sum(1 for _, c in RESULTS if c)
    print("\n" + str(len(fns) - failed) + "/" + str(len(fns)) + " functions, "
          + str(passed) + "/" + str(len(RESULTS)) + " checks passed")
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
