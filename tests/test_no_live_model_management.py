"""A test run must not move the operator's model.

test_llm_context_error.py drives the context-overflow path. It patches
llm.requests.post, so no HTTP leaves the process -- but the code under test
answers a context overflow by RELOADING the model (llm._try_heal_context ->
lmstudio.heal_context -> reload_via_cli -> `lms unload --all`), which would
unload the operator's model unless subprocess is patched too.

Two defences, because the first only protects suites that remember to ask:
  * offline_guard.no_model_management() raises if a suite reaches the CLI;
  * lmstudio refuses outright when F5_TEST_RUN is set, which run_all.py sets on
    every child -- so a suite written tomorrow is covered without opting in.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lmstudio as L  # noqa: E402
import offline_guard  # noqa: E402

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
        print("PASS  " + name)
    else:
        BAD += 1
        print("FAIL  " + name + ((": " + str(detail)) if detail else ""))


print("=" * 66)
print("THE RUNNER'S MARKER IS ENOUGH ON ITS OWN")
print("=" * 66)

_had = os.environ.get("F5_TEST_RUN")
os.environ["F5_TEST_RUN"] = "1"
try:
    ok, msg = L._lms_unload_all()
    check("`lms unload --all` is refused under a test run", ok is False, msg)
    check("...and the refusal says what to do instead",
          "Patch the boundary" in msg, msg)
    ok2, msg2 = L.reload_via_cli("some-model", 8192, 50)
    check("a CLI reload is refused too", ok2 is False, msg2)
    check("...before it can unload anything",
          "reload the model" in msg2, msg2)
finally:
    if _had is None:
        os.environ.pop("F5_TEST_RUN", None)
    else:
        os.environ["F5_TEST_RUN"] = _had

# Cleared explicitly: run_all.py sets F5_TEST_RUN for every child, so reading
# the ambient environment here would check nothing.
_had2 = os.environ.pop("F5_TEST_RUN", None)
try:
    check("outside a test run the guard steps aside",
          L._refuse_under_tests("x") is None)
finally:
    if _had2 is not None:
        os.environ["F5_TEST_RUN"] = _had2

# A suite that patched `lmstudio.subprocess` is exercising the CLI logic without
# touching the machine, and must keep working: the guard must not refuse
# before the patch can matter.
os.environ["F5_TEST_RUN"] = "1"
import types as _t
try:
    class _FakeSP:
        @staticmethod
        def run(*a, **k):
            class R:
                returncode = 0
                stdout = stderr = ""
            return R()
    _fc = _t.SimpleNamespace(deps=_t.SimpleNamespace(subprocess=_FakeSP))
    check("a handed-in subprocess is let through, since nothing can escape it",
          L._refuse_under_tests("x", ctx=_fc) is None)
    check("...and the unload helper actually runs its own logic",
          L._lms_unload_all(ctx=_fc) == (True, "ok"), L._lms_unload_all(ctx=_fc))
finally:
    if _had is None:
        os.environ.pop("F5_TEST_RUN", None)

print()
print("=" * 66)
print("AND THE EXPLICIT GUARD RAISES, RATHER THAN FAILING QUIETLY")
print("=" * 66)

_real_unload, _real_reload = L._lms_unload_all, L.reload_via_cli
offline_guard.no_model_management()
try:
    L._lms_unload_all()
    check("the guard stops an unload", False, "it did not raise")
except offline_guard.LiveModelTouched as exc:
    check("the guard stops an unload", True)
    check("...and names the CLI command that would have run",
          "lms unload --all" in str(exc), str(exc))
try:
    L.reload_via_cli("m", 4096, 50)
    check("the guard stops a reload", False, "it did not raise")
except offline_guard.LiveModelTouched:
    check("the guard stops a reload", True)

offline_guard.restore()
check("restore() puts the real functions back",
      L._lms_unload_all is _real_unload and L.reload_via_cli is _real_reload)
# restore() must not return early when offline_llm() was not installed, or the
# model guard would stay latched on for the rest of the process.
check("...even when the LLM guard was never installed",
      L._lms_unload_all is _real_unload)

print()
print("=" * 66)
print("THE SUITE THAT CAUSED THIS ASKS FOR THE GUARD")
print("=" * 66)

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_llm_context_error.py"), encoding="utf-8").read()
check("test_llm_context_error installs the model guard",
      "no_model_management()" in _src,
      "the suite that unloaded the operator's model no longer asks for the guard")
check("...before it imports llm",
      _src.index("no_model_management()") < _src.index("\nimport llm"), "order")

print()
print("%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)
