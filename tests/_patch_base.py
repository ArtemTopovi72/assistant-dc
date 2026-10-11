"""Shared save/rebind/restore context manager behind `_dr_patch` and `_image_patch`.

A subclass supplies `_apply(name, value)`: rebind `name` to `value` in every module
it considers a home for that name, and return the `(module, previous value)` pairs
it changed. The base undoes them in reverse order on exit.
"""


class RebindPatches:
    """Rebind module-level names for the duration of a `with` block."""

    def __init__(self, **kw):
        self.kw = kw
        self.undo = []          # (module, name, original)

    def _apply(self, name, value):
        raise NotImplementedError

    def __enter__(self):
        for name, value in self.kw.items():
            self.undo.extend((m, name, old) for m, old in self._apply(name, value))
        return self

    def __exit__(self, *exc):
        for mod, name, original in reversed(self.undo):
            setattr(mod, name, original)
        self.undo.clear()
        return False
