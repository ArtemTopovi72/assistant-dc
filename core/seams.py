"""The outside world a turn talks to, reachable through its context.

Code asks `text_model(ctx)`, `vision_model(ctx)` or `yes_no(ctx)` instead of importing
`llm` / `intent` itself; a context that carries its own (`ctx.text_model`, ...) gets that
one, every other context the real service. A suite then hands its fake in on the
context it builds -- it never writes over the module the next suite imports.
"""


def text_model(ctx):
    """`call_llm_simple(ctx, system, user, **kw) -> str`."""
    own = getattr(ctx, "text_model", None)
    if own is not None:
        return own
    import llm
    return llm.call_llm_simple


def vision_model(ctx):
    """`analyze_image_with_llm(ctx, **kw) -> str` (one picture or labelled `image_paths`)."""
    own = getattr(ctx, "vision_model", None)
    if own is not None:
        return own
    import llm
    return llm.analyze_image_with_llm


def yes_no(ctx):
    """`ask_yes(question, text, default=False, wait=True) -> bool`."""
    own = getattr(ctx, "yes_no", None)
    if own is not None:
        return own
    import intent
    return intent.ask_yes


def intent_read(ctx):
    """`read(ctx, text) -> dict`: the per-message intent read."""
    own = getattr(ctx, "intent_read", None)
    if own is not None:
        return own
    import intent
    return intent.read


def choice(ctx):
    """`ask_choice(question, text, options, default) -> str`: one of `options`."""
    own = getattr(ctx, "choice", None)
    if own is not None:
        return own
    import intent
    return intent.ask_choice
