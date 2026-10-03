"""Broadcast worker-stub helper for the GUI suites.

gui.py was decomposed into gui_workers / gui_common / gui_voice_tab /
gui_transfer_tab / gui_database_tab / gui_madhouse_tab / ... and the worker
classes no longer all live on `gui`.  Several suites still did
`getattr(gui, name)` from a NAME LIST, which has two failure modes:

  * AttributeError, when the class is gone from `gui` entirely; and
  * the SILENT one -- the name still exists on `gui`, so the patch "succeeds",
    but the actual call site reads it from the module it was imported into.
    The stub is then never consulted and the REAL QThread starts while the
    suite prints PASS.

The second one is not hypothetical.  test_gui_supplement11's
test_stop_vad_and_vad_ultra patched gui.TranscribeWorker, but
_on_vad_utterance lives in gui_voice_tab and reads gui_voice_tab's own
by-value binding.  A REAL TranscribeWorker was constructed and really started;
the test then dropped the reference, and Qt aborted the process with
0xC0000409 for destroying a QThread that was still running.

So do NOT maintain a hardcoded module list -- it goes stale exactly when a
class moves, which is the moment it needs to be right.  `_modules()` scans every
LOADED module that belongs to this repository, so a class can move anywhere and
still be found.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _modules():
    """Every loaded module whose source file lives inside this repository."""
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            if os.path.commonpath([os.path.abspath(f), _REPO]) == _REPO:
                yield mod
        except (ValueError, OSError):
            continue


def stub_workers(names):
    """Replace each named worker class with a start()-is-a-no-op subclass.

    Patches EVERY loaded repo module that binds the name, so no call site keeps
    reading the real class.  Returns the list of (module, name, previous)
    triples that were changed, for exact restoration.  Raises AttributeError if
    a name is bound nowhere, so a stale name list fails loudly instead of
    quietly doing nothing.
    """
    changed = []
    for n in names:
        mods = [m for m in _modules() if n in vars(m)]
        if not mods:
            raise AttributeError(
                "no loaded repo module binds %r -- the patch would be a no-op" % n)
        # Take the base from a module that holds the ORIGINAL class, i.e. one
        # whose binding is not already one of our stubs.
        base = None
        for m in mods:
            cand = getattr(m, n)
            if not getattr(cand, "_gui_patch_stub", False):
                base = cand
                break
        if base is None:
            base = getattr(mods[0], n)
        stub = type(n + "N", (base,), {"start": lambda self: None,
                                       "_gui_patch_stub": True})
        for m in mods:
            changed.append((m, n, getattr(m, n)))
            setattr(m, n, stub)
    return changed


def restore(changed):
    for mod, name, old in reversed(changed):
        setattr(mod, name, old)


class no_start:
    """Context manager form of `stub_workers`."""

    def __init__(self, *names):
        self.names = names
        self.changed = []

    def __enter__(self):
        self.changed = stub_workers(self.names)
        return self

    def __exit__(self, *a):
        restore(self.changed)
        return False
