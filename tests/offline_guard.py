"""Keep a suite off the real LM Studio.

Twenty suites that are meant to be offline were quietly calling
``localhost:1234``. Nothing depended on the answers — every one of them still
passed when the calls were blocked — but the cost was real:

  * a suite took 150 s and timed out instead of finishing in seconds, because
    each call retries three times against a server that is busy rendering;
  * results became machine-dependent, passing or failing on whether a model
    happened to be loaded;
  * a test run competed with the GPU for the very resource it was not supposed
    to touch.

``send_to_lm_studio`` returns ``None`` when the model is unavailable, and every
caller already has a fallback for that — it is the documented contract, not an
error path. So the stub here returns exactly ``None``: the suites see the state
they would see with LM Studio switched off, which is the state they were
implicitly relying on anyway.

Import it before the code under test makes its first call::

    import offline_guard; offline_guard.offline_llm()

Suites that genuinely exercise a live model (anything named ``*_live``,
``bench/lmstudio_harness_selftest``, ``test_agent_dialogue``) must NOT use this.
"""
import logging

logger = logging.getLogger(__name__)

_ORIGINAL = None
_ORIGINAL_UNLOAD = None
_ORIGINAL_RELOAD = None
CALLS: list = []          # (kind, n_messages) per intercepted call, for assertions


def offline_llm():
    """Point llm.send_to_lm_studio at the 'model unavailable' contract.

    Idempotent, and returns the number of calls intercepted so far so a suite
    can assert on it if it wants to.
    """
    global _ORIGINAL
    import llm

    if _ORIGINAL is not None:
        return CALLS

    _ORIGINAL = llm.send_to_lm_studio

    def _offline(ctx, messages=None, *a, **kw):
        CALLS.append(("send_to_lm_studio", len(messages or ())))
        return None

    llm.send_to_lm_studio = _offline

    # A module that did `from llm import send_to_lm_studio` before this ran keeps
    # its own binding, so patching llm alone would miss it. Only rebind modules
    # ALREADY imported — importing the rest here just to patch them would drag in
    # torch and Qt, and anything imported after this point picks up the stub from
    # llm anyway.
    import sys as _sys
    for mod in list(_sys.modules.values()):
        if getattr(mod, "__name__", "").startswith(("llm", "test")):
            continue
        if getattr(mod, "send_to_lm_studio", None) is _ORIGINAL:
            mod.send_to_lm_studio = _offline
    return CALLS


class LiveModelTouched(RuntimeError):
    """A suite tried to move the OPERATOR's model around."""


def no_model_management():
    """Forbid a suite from loading or unloading the machine's real model.

    Measured, not theoretical: tests/test_llm_context_error.py patched
    llm.requests.post, so no HTTP left the process -- but the code under test
    answers a context-overflow error by calling the context self-heal, which
    shells out to `lms unload --all` and then reloads. subprocess was never
    patched, so every full test run silently unloaded the operator's model.
    Downstream that looked like nothing at all: the next live deep-research run
    got empty answers from a server with no model and reported that the pages
    it had fetched "contained no facts relevant to the topic".

    Raising is deliberate. A silent stub would let the same class of mistake
    back in unnoticed; the point is that the suite is told.
    """
    global _ORIGINAL_UNLOAD, _ORIGINAL_RELOAD
    import lmstudio as _lms

    if _ORIGINAL_UNLOAD is not None:
        return

    _ORIGINAL_UNLOAD = _lms._lms_unload_all
    _ORIGINAL_RELOAD = _lms.reload_via_cli

    def _no_unload():
        raise LiveModelTouched(
            "a test tried to run `lms unload --all` on the real LM Studio. "
            "Patch the boundary you meant to exercise (llm._try_heal_context or "
            "lmstudio.reload_via_cli) instead of letting it reach the CLI.")

    def _no_reload(model_id, context_length, gpu_offload_pct, parallel=0):
        raise LiveModelTouched(
            "a test tried to reload the real LM Studio model (%r at %s tokens)."
            % (model_id, context_length))

    _lms._lms_unload_all = _no_unload
    _lms.reload_via_cli = _no_reload


def restore():
    """Put the real function back (for a suite that tests the guard itself)."""
    global _ORIGINAL, _ORIGINAL_UNLOAD, _ORIGINAL_RELOAD
    # The two guards are installed independently, so they are lifted
    # independently: an early return here left the model guard latched on.
    if _ORIGINAL is not None:
        import llm
        llm.send_to_lm_studio = _ORIGINAL
        _ORIGINAL = None
    if _ORIGINAL_UNLOAD is not None:
        import lmstudio as _lms
        _lms._lms_unload_all = _ORIGINAL_UNLOAD
        _lms.reload_via_cli = _ORIGINAL_RELOAD
        _ORIGINAL_UNLOAD = _ORIGINAL_RELOAD = None
