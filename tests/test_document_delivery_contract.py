"""Every producer of a deliverable file must speak the sender's word.

tg_tasks sends `document_path` only when `document_status` reads exactly
"success". pack_archive wrote "ok". So every archive the coding agent built was
dropped in silence -- and the tool's own return value said "it will be sent to
the user", so the model told the user it had been. Nothing anywhere noticed:

  * the unit suites stub the delivery layer, so they never compare the two;
  * bench/sandbox_e2e checked that a file existed at document_path and never
    the status, so it reported "delivered Thief_modified.jar" for a file
    Telegram would have thrown away;
  * the honesty correction that catches an undelivered file only arms for
    presentation requests, by prefix, so a packed archive had no backstop.

A string agreed on in two files and checked in neither is not a contract. This
suite reads both sides out of the source and compares them.

Run: venv/Scripts/python.exe tests/test_document_delivery_contract.py
"""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


def _assigned_values(src, key):
    """Every literal assigned to state["<key>"] at any depth."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.slice, ast.Constant)
                    and tgt.slice.value == key
                    and isinstance(node.value, ast.Constant)):
                out.append(node.value.value)
    return out


def _accepted_value(src, key):
    """The literal the READER compares that key against."""
    found = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Call)
                and isinstance(left.func, ast.Attribute)
                and left.func.attr == "get"
                and left.args and isinstance(left.args[0], ast.Constant)
                and left.args[0].value == key):
            continue
        for comp in node.comparators:
            if isinstance(comp, ast.Constant):
                found.add(comp.value)
    return found


def test_writers_and_reader_agree_on_document_status():
    writers = {}
    for mod in ("agent/tools.py", "agent/tool_code_handlers.py"):
        for value in _assigned_values(_read(mod), "document_status"):
            writers.setdefault(value, []).append(mod)
    accepted = _accepted_value(_read("bot/tg_tasks.py"), "document_status")

    check("the reader was found", accepted, "no comparison on document_status")
    check("writers were found", writers, "nothing assigns document_status")
    bad = {v: m for v, m in writers.items() if v not in accepted}
    check("every writer uses a value the sender accepts", not bad,
          "sender accepts %s; written but never sent: %s" % (sorted(accepted), bad))


def test_pack_archive_promises_only_what_it_delivers():
    """The tool tells the model the file WILL be sent. That has to be true."""
    src = _read("agent/tool_code_handlers.py")
    body = src[src.index("def _handle_pack_archive"):]
    body = body[:body.index("\ndef ")] if "\ndef " in body[10:] else body
    promises = "will be sent to the user" in body
    sets_status = 'state["document_status"] = "success"' in body
    check("pack_archive promises delivery", promises,
          "the promise moved; re-anchor this check")
    check("...and sets the status that causes it", sets_status,
          "pack_archive promises a delivery it does not arm")


def test_the_bench_checks_the_status_too():
    """A bench that only checks the path cannot see this class of defect."""
    src = _read(os.path.join("bench", "sandbox_e2e.py"))
    check("the e2e bench asserts document_status",
          "document_status" in src,
          "bench/sandbox_e2e.py reports delivery without checking the status "
          "the delivery layer requires")


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
    print("\n%d/%d functions, %d/%d checks passed"
          % (len(fns) - failed, len(fns), passed, len(RESULTS)))
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
