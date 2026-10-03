"""Stage 2 · control plane: the producer-consumer prerequisite graph and the
semantic argument gate in tool_graph, plus their wiring into tools.execute_tool.

What this pins down:

  * A tool that consumes a resource nobody has produced yet is refused BEFORE
    its handler runs, with a message that names the missing prerequisite and
    the tool that produces it. The agent loop replans on the "[TOOL ERROR]"
    prefix, so an unactionable message costs a wasted round.
  * transfer_image needs TWO pictures, not one — the single working image is
    not enough, and saying so is what stops the model from retrying forever.
  * Arguments that are the right TYPE but the wrong MEANING are refused:
    an effectively-empty prompt (this project's default-person landmine), a
    redraw mode that does not exist (pydantic silently coerces it to None and
    the requested mode is lost), prose handed to the calculator, and unknown
    transfer roles.

Hermetic: no LM Studio, no ComfyUI, no GPU, no network. Every check runs
against plain dicts and a temp PNG.

Run: venv/Scripts/python.exe -m pytest tests/test_tool_prereq_graph.py -q
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import logging
logging.basicConfig(level=logging.CRITICAL)

import tool_graph as tg


class _Ctx:
    """Minimal stand-in for the real Context: the gate only reads these."""
    def __init__(self, last_image_path=None, reference_images=None,
                 web_search_enabled=True):
        self.last_image_path = last_image_path
        self.reference_images = list(reference_images or [])
        self.web_search_enabled = web_search_enabled


_TMP = []


def _png(name):
    """A real file on disk — the probe rejects paths that do not exist, and a
    test that used a made-up path would pass for the wrong reason."""
    d = tempfile.mkdtemp(prefix="tgprereq_")
    p = os.path.join(d, name)
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
    _TMP.append(p)
    return p


# --- the graph tables themselves --------------------------------------------

def test_graph_tables_are_coherent():
    """Every required/produced resource must exist, and every producer named in
    an error message must be a real registered tool. A typo here would ship an
    error telling the model to call a tool that does not exist."""
    import tools as tmod
    known = set(tmod._BY_NAME)
    for tool, reqs in tg.REQUIRES.items():
        assert tool in known, f"REQUIRES names unknown tool {tool!r}"
        for r in reqs:
            assert r in tg.RESOURCES, f"{tool} requires unknown resource {r!r}"
    for tool, prods in tg.PRODUCES.items():
        assert tool in known, f"PRODUCES names unknown tool {tool!r}"
        for r in prods:
            assert r in tg.RESOURCES, f"{tool} produces unknown resource {r!r}"
    for res in tg.RESOURCES.values():
        for p in res.producers:
            assert p in known, f"resource {res.name} lists unknown producer {p!r}"
            assert res.name in tg.PRODUCES.get(p, ()), \
                f"{p} is listed as a producer of {res.name} but PRODUCES disagrees"


def test_every_image_edit_tool_requires_an_image():
    """The whole point of the graph: no image-consuming tool may be omitted."""
    for tool in ("redraw_image", "inpaint_image", "inspect_image",
                 "fix_hands", "fix_artifact"):
        assert "image" in tg.REQUIRES.get(tool, ()), f"{tool} lost its image precondition"
    assert "two_images" in tg.REQUIRES.get("transfer_image", ())


# --- preconditions ----------------------------------------------------------

def test_missing_image_names_the_producing_tool():
    ctx, state = _Ctx(), {}
    msg = tg.missing_prerequisite(ctx, state, "inpaint_image")
    assert msg and msg.startswith("[TOOL ERROR]"), msg
    assert "generate_image" in msg and "find_photo" in msg, \
        f"message does not name the producing tools: {msg!r}"


def test_present_image_satisfies_the_precondition():
    p = _png("a.png")
    assert tg.missing_prerequisite(_Ctx(last_image_path=p), {}, "inpaint_image") is None
    # and via state, which is the path the graph actually fills in
    assert tg.missing_prerequisite(_Ctx(), {"image_path": p}, "fix_hands") is None


def test_dangling_path_is_not_an_image():
    """A path recorded in state but deleted from disk must NOT satisfy the
    precondition — otherwise the gate passes and the handler fails deeper."""
    ctx = _Ctx(last_image_path=os.path.join(tempfile.gettempdir(), "_tg_missing_zz.png"))
    assert tg.missing_prerequisite(ctx, {}, "redraw_image") is not None


def test_transfer_image_needs_two_pictures():
    one = _png("one.png")
    ctx = _Ctx(last_image_path=one)
    msg = tg.missing_prerequisite(ctx, {}, "transfer_image")
    assert msg and "TWO" in msg, f"one image wrongly accepted for transfer: {msg!r}"
    ctx.reference_images = [_png("two.png")]
    assert tg.missing_prerequisite(ctx, {}, "transfer_image") is None


def test_web_tools_blocked_when_internet_is_off():
    off = _Ctx(web_search_enabled=False)
    for tool in ("search", "deep_research", "find_photo"):
        msg = tg.missing_prerequisite(off, {}, tool)
        assert msg and msg.startswith("[TOOL ERROR]"), f"{tool} ran with internet off"
    on = _Ctx(web_search_enabled=True)
    assert tg.missing_prerequisite(on, {}, "search") is None


# --- semantic argument validation -------------------------------------------

def test_empty_and_placeholder_prompts_are_refused():
    """The default-person landmine: ' . ' and 'none' are valid non-empty strings
    that pydantic's min_length=1 accepts and the renderer turns into a picture
    of a stranger."""
    for bad in ("   ", ".", "...", "none", "N/A", "-", "?", "прompt" [:1]):
        msg = tg.invalid_arguments("generate_image", {"description": bad})
        assert msg and msg.startswith("[TOOL ERROR]"), f"empty prompt accepted: {bad!r}"
    assert tg.invalid_arguments("generate_image", {"description": "a red fox"}) is None
    assert tg.invalid_arguments("generate_image", {"description": "рыжий кот"}) is None


def test_absent_field_is_left_to_pydantic():
    """The gate must not duplicate the required-field error — pydantic's message
    carries the field description the model needs."""
    assert tg.invalid_arguments("generate_image", {}) is None


def test_redraw_mode_must_be_a_real_mode():
    msg = tg.invalid_arguments("redraw_image", {"mode": "sharpen", "instructions": "x"})
    assert msg and "sharpen" in msg
    for mode in tg.REDRAW_MODES:
        assert tg.invalid_arguments("redraw_image", {"mode": mode}) is None
    assert tg.invalid_arguments("redraw_image", {"instructions": "make it snowy"}) is None


def test_calculate_rejects_prose_but_accepts_math():
    for prose in ("how many people live in Tokyo",
                  "the population of France divided by two",
                  "x + y"):
        msg = tg.invalid_arguments("calculate", {"expression": prose})
        assert msg and msg.startswith("[TOOL ERROR]"), f"prose accepted: {prose!r}"
    for expr in ("1234*5678", "(15*1.2)/3", "sqrt(144)", "log(1000, 10)", "pi*2"):
        assert tg.invalid_arguments("calculate", {"expression": expr}) is None, expr


def test_transfer_roles_are_validated():
    msg = tg.invalid_arguments(
        "transfer_image", {"instructions": "put the hat on him", "roles": ["hat_thing"]})
    assert msg and "hat_thing" in msg
    assert tg.invalid_arguments(
        "transfer_image", {"instructions": "put the hat on him",
                           "roles": ["object_source"]}) is None


def test_precondition_wins_over_argument_complaint():
    """With no picture at all, 'there is nothing to edit' is the honest error —
    complaining about the region argument would send the model down a dead end."""
    msg = tg.check_tool_call(_Ctx(), {}, "inpaint_image",
                             {"instructions": ".", "region": "."})
    assert "needs a picture to work on" in msg, msg


# --- the wiring -------------------------------------------------------------

def test_execute_tool_gates_before_the_handler_runs():
    """The gate must fire inside execute_tool, not merely exist as a library."""
    import tools as tmod

    called = []
    spec = tmod._BY_NAME["fix_hands"]
    patched = tmod.ToolSpec(spec.name, spec.schema,
                            lambda c, s, a: called.append(1) or "ok", spec.args_model)
    tmod._BY_NAME["fix_hands"] = patched
    try:
        out = tmod.execute_tool(_Ctx(), {}, "fix_hands", {})
        assert out.startswith("[TOOL ERROR]") and "generate_image" in out, out
        assert not called, "handler ran despite a missing prerequisite"
        # with an image present the same call reaches the handler
        out2 = tmod.execute_tool(_Ctx(last_image_path=_png("h.png")), {}, "fix_hands", {})
        assert out2 == "ok" and called, out2
    finally:
        tmod._BY_NAME["fix_hands"] = spec


def test_execute_tool_blocks_a_placeholder_prompt():
    import tools as tmod
    called = []
    spec = tmod._BY_NAME["generate_image"]
    tmod._BY_NAME["generate_image"] = tmod.ToolSpec(
        spec.name, spec.schema, lambda c, s, a: called.append(1) or "drew it",
        spec.args_model)
    try:
        out = tmod.execute_tool(_Ctx(), {}, "generate_image", {"description": " . "})
        assert out.startswith("[TOOL ERROR]"), out
        assert not called, "renderer ran on an effectively empty prompt"
    finally:
        tmod._BY_NAME["generate_image"] = spec


def test_execute_tool_gate_survives_a_junk_ctx():
    """Telegram/GUI hand in different ctx objects; a gate that raises would turn
    every tool call into a crash. It must degrade to 'no resource'."""
    import tools as tmod
    out = tmod.execute_tool(object(), {"image_path": 12345}, "inspect_image",
                            {"check": "is it a cat?"})
    assert out.startswith("[TOOL ERROR]"), out


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\ndone")
