"""Broadcast monkeypatch helper for the image pipeline.

image.py was split into image_sizing / image_grounding / image_masks / ... and
each of those modules imports the shared helpers BY VALUE.  A patch applied to
`image.X` alone therefore only moves the callers that still live in image.py —
the classic split-brain seam.  `Patches` sets the attribute on every module that
currently binds that name, and restores exactly the bindings it changed.

Usage:
    from _image_patch import Patches, broadcast
    with Patches(_upload_image_to_comfy=lambda *a, **k: "u.png"):
        ...
"""
import sys

# Every module that may hold a binding of a patched image-pipeline name.
_TARGET_MODULES = (
    "image", "image_sizing", "image_grounding", "image_masks", "image_maskqa",
    "image_handfix", "image_identity", "image_transfer", "image_router",
    "image_contained", "image_contained_graphs", "image_contained_firered",
    "image_transforms", "image_objects", "image_generate",
    "comfy_client", "compositing",
)


def _modules():
    for name in _TARGET_MODULES:
        mod = sys.modules.get(name)
        if mod is not None:
            yield mod


def broadcast(name, value):
    """Set `name` to `value` on every loaded module that already binds it.

    Returns the list of (module, previous value) pairs that were changed.
    """
    changed = []
    for mod in _modules():
        if hasattr(mod, name):
            changed.append((mod, getattr(mod, name)))
            setattr(mod, name, value)
    if not changed:
        raise AttributeError(
            "no loaded image module binds %r — the patch would be a no-op" % name)
    return changed


class Patches:
    """Context manager applying `broadcast` to each keyword, then undoing it."""

    def __init__(self, **kw):
        self.kw = kw
        self.undo = []

    def __enter__(self):
        for k, v in self.kw.items():
            self.undo.extend((m, k, old) for m, old in broadcast(k, v))
        return self

    def __exit__(self, *a):
        for mod, name, old in reversed(self.undo):
            setattr(mod, name, old)
        self.undo = []
        return False
