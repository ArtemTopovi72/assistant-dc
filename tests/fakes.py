"""Stand-ins handed to the code under test, never written over a module.

`Over(base, **overrides)` answers the overrides and falls through to `base` for the rest,
so a suite gets the real `video` module with a fake renderer in one object and nothing
global changes: the next suite in the same process sees the real module.
"""


class Over:
    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)
