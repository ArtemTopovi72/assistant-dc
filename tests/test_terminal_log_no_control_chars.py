"""The GUI log pane must never receive ANSI escapes or control characters.

Live 2026-09-13: after a photo made LM Studio reload the model, the
"model revive ok" record carried the raw `lms load` TUI (ESC[?25l plus a
braille spinner). QTextEdit.append crashed the whole app with an access
violation inside DirectWrite (faulthandler: gui._append_log). The bridge
now cleans every line, and llm keeps only the last plain line of lms output.
"""
import os, sys, logging, inspect
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import gui_log_bridge as B

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

RAW = "\x1b[?25l\nLoading gemma ⠙\x1b[?25l\nLoading gemma ⠹\rLoading gemma ⠸\x1b[?25h\nModel loaded.\x00\x07"
out = B.clean_terminal_text(RAW)
check("ANSI escapes are gone", "\x1b" not in out, repr(out))
check("control characters are gone", all(ord(c) >= 32 or c == "\n" for c in out), repr(out))
check("braille spinner glyphs are gone", not any("⠀" <= c <= "⣿" for c in out), repr(out))
check("a CR-driven progress line keeps its last frame", "Loading gemma" in out and out.count("Loading gemma") == 2, repr(out))
check("plain text survives", "Model loaded." in out, repr(out))
check("ordinary lines are untouched", B.clean_terminal_text("12:00:01  assistant: hello — «ок»") == "12:00:01  assistant: hello — «ок»")

# the handler applies it
got = []
h = B.QtLogHandler()
h.bridge.line.connect(got.append)
rec = logging.LogRecord("t", logging.WARNING, __file__, 1, "revive: %s", ("\x1b[?25l⠙ ok",), None)
h.emit(rec)
check("the handler emits cleaned text", got and "\x1b" not in got[0] and "⠙" not in got[0] and "ok" in got[0], got)

import llm as L
src = inspect.getsource(L)
check("llm strips the lms TUI before logging the revive message", "lines[-1] if lines" in src and "[\\r\\n]+" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
