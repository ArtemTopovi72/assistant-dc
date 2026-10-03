"""Mutation harness: prove the cancel-isolation and broadcast suites are load-bearing.

Each mutant restores exactly one pre-fix behaviour. A mutant that SURVIVES means the
new test would not have caught the original bug.

Run: PYTHONIOENCODING=utf-8 venv/Scripts/python.exe tests/_mut_cancel_bcast.py
"""
import os, sys, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

Q = chr(34)

MUTANTS = [
    # (label, file, find, replace, suite)
    ("shutdown ignores the subscription (the original bug)",
     "bot/tg_bot.py",
     'if "startup" not in (user.subscriptions or []):\n                continue',
     'pass',
     "tests/test_tg_broadcast.py"),

    ("shutdown notice back to hardcoded English",
     "bot/tg_bot.py",
     '_t("going_offline", self._lang(sess)),',
     Q + "\U0001F534 <b>Going offline.</b> I'll notify you when I'm back." + Q + ",",
     "tests/test_tg_broadcast.py"),

    ("startup notice back to hardcoded English",
     "bot/tg_bot.py",
     '_t("back_online", self._lang(sess)),',
     Q + "\U0001F7E2 <b>Hello, I'm back online!</b> Ready to assist." + Q + ",",
     "tests/test_tg_broadcast.py"),

    # The `or []` in the broadcasts and _User.__post_init__ are deliberately
    # redundant, so removing EITHER alone is an equivalent mutant — the other one
    # still holds. Remove both, which is the regression that actually matters.
    ("both None guards removed (a JSON-null row reaches the broadcast)",
     "bot/tg_bot.py",
     [('if "startup" in (user.subscriptions or []):',
       'if "startup" in user.subscriptions:'),
      ('if "startup" not in (user.subscriptions or []):',
       'if "startup" not in user.subscriptions:'),
      ("if not isinstance(self.subscriptions, list):", "if False:")],
     None,
     "tests/test_tg_broadcast.py"),

    ("the bot gets the raw ctx again (the original bug)",
     "gui/gui.py",
     "        view = self._bot_ctx_view\n",
     "        return ctx\n        view = self._bot_ctx_view\n",
     "tests/test_tg_cancel_isolation.py"),

    ("the constructor bypasses the scoped view",
     "gui/gui.py",
     "get_ctx         = self._bot_ctx,",
     "get_ctx         = lambda: getattr(self.host, 'ctx', None),",
     "tests/test_tg_cancel_isolation.py"),

    ("the view is rebuilt on every call (identity churn)",
     "gui/gui.py",
     'if view is None or object.__getattribute__(view, "_ctx") is not ctx:',
     "if True:",
     "tests/test_tg_cancel_isolation.py"),

    ("a fresh cancel token on every rebuild",
     "gui/gui.py",
     "view = _ScopedCtx(ctx, self._bot_cancel)",
     "view = _ScopedCtx(ctx, threading.Event())",
     "tests/test_tg_cancel_isolation.py"),
]

env = dict(os.environ, PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="offscreen",
           USE_GUI="0")

caught = survived = skipped = 0
for label, fname, find, repl, suite in MUTANTS:
    path = os.path.join(ROOT, fname)
    with open(path, "r", encoding="utf-8") as fh:
        original = fh.read()
    edits = find if isinstance(find, list) else [(find, repl)]
    bad = [f for f, _ in edits if original.count(f) != 1]
    if bad:
        print(f"SKIP (anchor missing x{len(bad)})  {label}")
        skipped += 1
        continue
    mutated = original
    for f, rep in edits:
        mutated = mutated.replace(f, rep, 1)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(mutated)
        r = subprocess.run([sys.executable, suite], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=300)
    finally:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(original)
    if r.returncode != 0:
        caught += 1
        fails = [l for l in (r.stdout or "").splitlines() if l.startswith("FAIL")]
        why = fails[0][6:].strip() if fails else "(crashed)"
        print(f"CAUGHT    {label}\n            by: {why}")
    else:
        survived += 1
        print(f"SURVIVED  {label}   <-- the test would not have caught this")

print()
print(f"{caught} caught, {survived} survived, {skipped} skipped "
      f"({len(MUTANTS)} mutants)")
sys.exit(1 if survived or skipped else 0)
