"""The `_image` seam shared by image_contained and image_contained_firered.

This module used to also hold three crop-and-inpaint workflow builders
(the old model, BrushNet, FLUX.1-Fill). All three were removed from the product --
FireRed is the only edit engine -- so only the proxy is left.

`_ImageProxy` and the `_image` instance live here rather than in both files. A
second proxy object would work today -- it is stateless -- but two definitions
of the seam that exists to keep bindings singular is the wrong shape. image_contained
imports both back.

The seam note from image_contained applies unchanged: everything from image.py is
reached through `_image.<name>` so it resolves at CALL time. Binding those names
by value here would give this module its own copy, and a suite patching
`image.<name>` would move only half the behaviour while still printing PASS.
"""


class _ImageProxy:
    """Attribute proxy onto the still-monolithic image.py.

    Reading through it defers the import to call time (no import cycle) and
    always resolves the CURRENT binding, so a runtime patch of image.<name> is
    honoured here even though the caller has moved out of image.py.
    """

    def __getattr__(self, name):
        import image
        return getattr(image, name)


_image = _ImageProxy()
