"""The image half of the delivery contract, and the hole next to it.

Companion to test_document_delivery_contract. tg_tasks sends state
["image_path"] only when state["image_status"] is set and is not "fail",
"error" or "". So every tool that PRODUCES a picture has to set that key, and
any word outside the accepted set silently drops the result — which is exactly
how pack_archive's "ok" lost every archive it ever built.

It also pins the opposite rule, which is easier to break by trying to be
helpful: open_image must NOT set the status. It is an input selector — it
points the vision tools at a file already sitting in the working folder. Give
it a delivery status and the bot posts the user's own screenshot back at them
every time it looks at one.

KNOWN GAP, deliberately recorded rather than asserted:
a picture the agent CREATES in the sandbox (run_code writing chart.png) has no
route to the user at all. open_image sets image_path with no status, so
delivery skips it, and the honesty correction then tells the user the picture
could not be delivered. The bot stays truthful and the user still gets no
chart. Closing that needs a tool that hands a sandbox file over; until it
exists, test_a_created_picture_has_no_delivery_route holds the shape of the
hole so it cannot be forgotten or "fixed" by loosening the gate.

Run: venv/Scripts/python.exe tests/test_image_delivery_contract.py
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The words tg_tasks refuses. Read from the source below rather than trusted.
REJECTED = ("fail", "error", "")

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


def test_the_gate_still_rejects_the_words_this_suite_assumes():
    """Read the refusal list out of tg_tasks instead of trusting a comment."""
    src = _read("bot/tg_tasks.py")
    check("the image gate is where it was",
          'img_status not in ("fail", "error", "")' in src,
          "the delivery condition moved; re-anchor this suite before trusting it")


def test_every_producer_sets_a_status_that_delivers():
    values = {}
    for mod in ("agent/tool_image_handlers.py", "agent/tools.py"):
        for v in _assigned_values(_read(mod), "image_status"):
            values.setdefault(v, []).append(mod)
    check("producers were found", values, "nothing assigns image_status")
    bad = {v: m for v, m in values.items() if v in REJECTED}
    check("no producer writes a word the gate drops", not bad,
          "written but never sent: %s" % bad)


def test_open_image_stays_an_input_selector():
    """It points the vision tools at a file; it does not hand one over."""
    src = _read("agent/tool_code_handlers.py")
    body = src[src.index("def _handle_open_image"):]
    end = body.index("\n@_guard")
    body = body[:end]
    check("open_image sets the path", 'state["image_path"]' in body)
    check("open_image does NOT set a delivery status",
          "image_status" not in body,
          "open_image would echo every picture the user uploads back at them")


def test_find_content_is_the_sandbox_delivery_route():
    """The hole is closed: find_content (2026-09-14) delivers the first match.

    It sets image_status to a value the gate accepts, so a found photo reaches
    the chat; open_image still must not (see the test above).
    """
    src = _read("agent/tool_code_handlers.py")
    body = src[src.index("def _handle_find_content"):]
    body = body[:body.index("\n@_guard")]
    check("find_content sets the path", 'state["image_path"]' in body)
    check("find_content sets a delivery status the gate accepts",
          'state["image_status"] = "ok"' in body)
