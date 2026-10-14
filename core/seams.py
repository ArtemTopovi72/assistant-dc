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


def chat_model(ctx):
    """`send_to_lm_studio(ctx, messages, tools=..., tool_choice=..., **kw) -> message dict`."""
    own = getattr(ctx, "chat_model", None)
    if own is not None:
        return own
    import llm
    return llm.send_to_lm_studio


def run_tool(ctx):
    """`execute_tool(ctx, state, tool_name, tool_args) -> str`."""
    own = getattr(ctx, "run_tool", None)
    if own is not None:
        return own
    import tools
    return tools.execute_tool


def file_dialog(widget):
    """The `QFileDialog` a window opens: `getOpenFileName(s)`, `getSaveFileName`, `getExistingDirectory`."""
    own = getattr(widget, "file_dialog", None)
    if own is not None:
        return own
    from PyQt5.QtWidgets import QFileDialog
    return QFileDialog


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


class Over:
    """`base` with some names answered differently: the overrides, then `base` for the
    rest. A run's own settings over the defaults, or a suite's fake renderer over the
    real module -- one object handed in, nothing global written."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


def part(ctx, holder: str, name: str, default):
    """`default`, unless the context's `holder` (a namespace of stand-ins, e.g.
    `ctx.research_parts`) names its own `name`: one pipeline stage swapped for a run,
    every other stage the real one."""
    own = getattr(ctx, holder, None)
    got = vars(own).get(name) if own is not None and hasattr(own, "__dict__") else None
    return got if got is not None else default


def dep(ctx, name: str, default):
    """`default` (the real function, module or client), or the stand-in a context
    carries for it in `ctx.deps`: a suite fakes one dependency of one call."""
    return part(ctx, "deps", name, default)


def module(owner, name: str):
    """The module `name`, unless `owner` (a context, or a widget -- its parents are asked
    too) carries its own in `owner.modules`: a suite hands in a stand-in for one import.
    `owner` may also be that dict itself, and a `None` entry stands for a module that is
    not installed (the import raises)."""
    w = owner
    while w is not None:
        own = w if isinstance(w, dict) else getattr(w, "modules", None)
        if own is not None and name in own:
            if own[name] is None:
                raise ImportError(name)
            return own[name]
        up = None if isinstance(w, dict) else getattr(w, "parent", None)
        w = up() if callable(up) else None
    import importlib
    return importlib.import_module(name)


def widget_part(owner, name: str, default):
    """`default`, unless `owner` (a widget -- its parents are asked too) carries its own
    `name` in `owner.parts`: a suite hands in a stand-in dialog or worker for one window."""
    w = owner
    while w is not None:
        own = getattr(w, "parts", None)
        if isinstance(own, dict) and own.get(name) is not None:
            return own[name]
        up = getattr(w, "parent", None)
        w = up() if callable(up) else None
    return default
