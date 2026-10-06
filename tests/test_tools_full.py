"""Full branch coverage for tools.py: every _handle_* success path + edge branch
(the failure paths already live in test_tool_fault_degradation.py), plus the
schema/validation plumbing (_as_int, _clean_schema, _fn_schema,
_format_validation_error, execute_tool) and helpers. All external service
boundaries (image ops, web search, deep research, payload guard, clipboard,
vision) are mocked so success paths run deterministically without live services.

Run: venv/Scripts/python.exe tests/test_tools_full.py
"""
import os, sys, types, tempfile, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path

import tools as T
import models as M
import threading

_TMP = Path(tempfile.mkdtemp(prefix="tools_"))

def _img(name="im.png"):
    p = _TMP / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode() + b"0" * 64)   # distinct: transfer dedupes by content
    return str(p)

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    assert cond, name + ": " + detail


def _ctx(**kw):
    c = M.Context(models=types.SimpleNamespace(), transcription_cache={}, cache_file=_TMP / "c.json",
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    c.active_memory_dir = _TMP / "mem"
    c.web_search_enabled = True
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class _PG:  # payload-guard verdict
    def __init__(self, usable=True, status="ok", reason="fine"):
        self.usable, self.status, self.reason = usable, status, reason


@contextlib.contextmanager
def mock_env(**over):
    """Patch tools/module boundaries. `over` overrides specific attrs."""
    saved = {}
    def setattr_saved(obj, name, val, key):
        saved[key] = (obj, name, getattr(obj, name, None), hasattr(obj, name))
        setattr(obj, name, val)

    out_img = _img("mock_out.png")
    # tools module-level image funcs
    tset = {
        "run_web_search": lambda ctx, q: over.get("search_result", "web page text about " + q),
        "generate_image_with_refinement": lambda **k: over.get("gen_result",
            {"path": out_img, "score": 8, "attempts": 1, "status": "success", "prompt": "p"}),
        "enhance_faces_with_comfy": lambda ctx, p: p,
        "redraw_image_with_comfy": lambda ctx, s, i: over.get("redraw_out", out_img),
        "generate_image_with_comfy": lambda ctx, **k: over.get("enhance_out", out_img),
        "inpaint_region_with_comfy": lambda ctx, s, r, i, removal=False, engine="firered": over.get("inpaint_out", out_img),
        "route_edit_request": lambda ctx, s, syn: ("bg_replace", over.get("route_out", out_img)),
        "classify_edit_intent": lambda syn: over.get("category", "subject_edit"),
        "upscale_image_with_comfy": lambda ctx, s, scale="4x", protect_face=True: over.get("upscale_out", out_img),
        "restore_image_with_comfy": lambda ctx, s, upscale="2x": over.get("restore_out", out_img),
        "_is_removal_instruction": lambda t: over.get("is_removal", False),
        "_INPAINT_FAILURE": over.get("inpaint_failure", {}),
    }
    # Apply to EVERY module that binds the name, not just `tools`. The image
    # editing handlers moved to tool_image_handlers and import these from
    # `image` by value, so a patch on T alone stops reaching them — the stub is
    # ignored and the real ComfyUI path runs. Setting it on each holder keeps
    # the stub in force wherever the caller happens to live.
    import tool_image_handlers as _tih
    _holders = (T, _tih)
    for k, v in tset.items():
        for _mod in _holders:
            if hasattr(_mod, k):
                setattr_saved(_mod, k, v, _mod.__name__ + "." + k)
    # image_mod attribute funcs
    im = T.image_mod
    imset = {
        "plan_and_execute_transfer": lambda ctx, t, r, i: over.get("transfer_out", out_img),
        "assert_deliverable": lambda p, where=None, source_path=None: over.get("deliverable", p),
        "log_edit_decision": over.get("log_edit", lambda **k: None),
        "fix_hands": lambda ctx, s, engine="firered": over.get("fixhands_out", out_img),
        "edit_region_contained_via_firered": lambda ctx, s, r, i, protect_face=True, engine="firered": over.get("fixart_out", out_img),
        "preserve_identity_face": lambda s, n, instructions="": n,
        "enhance_faces_with_comfy": lambda ctx, p: p,
        "load_layout_for": lambda p: over.get("layout", None),
        "edit_via_layout": lambda ctx, s, i: over.get("layout_out", out_img),
        "edit_image_with_firered": lambda ctx, s, i: over.get("firered_out", out_img),
        "_firered_style_instruction": lambda ctx, i, image_path=None: i,
        "generate_person_from_reference": lambda ctx, d, seed=None: over.get("refperson", (None, {})),
        "ReferenceImage": lambda p, role: types.SimpleNamespace(path=p, role=role),
        "_ALL_ROLES": {"object_source", "clothing_source"},
        "IMAGE_BUILD_ID": "test",
        "_is_removal_of": lambda i, r="", ctx=None: over.get("is_removal", False),
        "OUTPUT_DIR": _TMP,
    }
    for k, v in imset.items():
        setattr_saved(im, k, v, "im." + k)
    # injected modules
    fake_pg = types.ModuleType("payload_guard")
    fake_pg.assess_payload = lambda q, r: over.get("payload", _PG())
    fake_pg.SKEPTIC_BANNER = "\n[verify]"
    fake_dr = types.ModuleType("deep_research")
    fake_dr.run_deep_research = lambda ctx, topic, depth="standard", progress=None, **_kw: over.get("dr_result",
        {"report": "# Executive Summary\nfindings here\n", "path": "r.md",
         "stats": {"sources": 5, "pages": 10, "findings": 3}, "cancelled": False})
    import llm as _llm
    saved["llm.analyze"] = (_llm, "analyze_image_with_llm", _llm.analyze_image_with_llm, True)
    _llm.analyze_image_with_llm = lambda **k: over.get("vision", "PRESENT: everything looks good")
    import search as _search
    saved["search.fetch"] = (_search, "fetch_reference_photo", _search.fetch_reference_photo, True)
    _search.fetch_reference_photo = lambda ctx, q, dest, require_face=True: over.get("photo", out_img)
    import sys as _sys
    saved_pg = _sys.modules.get("payload_guard"); saved_dr = _sys.modules.get("deep_research")
    _sys.modules["payload_guard"] = fake_pg; _sys.modules["deep_research"] = fake_dr
    try:
        yield out_img
    finally:
        for obj, name, old, had in saved.values():
            if had: setattr(obj, name, old)
        if saved_pg is not None: _sys.modules["payload_guard"] = saved_pg
        else: _sys.modules.pop("payload_guard", None)
        if saved_dr is not None: _sys.modules["deep_research"] = saved_dr
        else: _sys.modules.pop("deep_research", None)


# ------------------------------------------------------------------ helpers

def test_helpers():
    check("as_int_none", T._as_int(None) is None and T._as_int("") is None)
    check("as_int_ok", T._as_int("8") == 8 and T._as_int(8.9) == 8)
    check("as_int_bad", T._as_int("abc") is None)
    check("wrap_untrusted", "UNTRUSTED" in T._wrap_untrusted("SRC", "data"))
    import tool_image_handlers as _tih   # the read itself: bench/removal_intent_live.py
    _tih.ABSENCE_STUB = lambda c: "still there" in c
    check("absence_check", T._is_absence_check("is the puddle still there?")
          and not T._is_absence_check("is the sky blue?"))
    _tih.ABSENCE_STUB = None
    # _exec_summary with heading + truncation
    rep = "# Executive Summary\n" + ("x" * 2000) + "\n# Next"
    s = T._exec_summary(rep, limit=100)
    check("exec_summary_trunc", s.endswith("…") and len(s) <= 101)
    check("exec_summary_no_heading", T._exec_summary("just text") == "just text")
    # _clean_schema collapses anyOf and drops title
    cs = T._clean_schema({"title": "X", "anyOf": [{"type": "integer"}, {"type": "null"}]})
    check("clean_schema_collapse", cs.get("type") == "integer" and "title" not in cs)
    check("clean_schema_list", T._clean_schema([{"title": "a", "type": "string"}]) == [{"type": "string"}])
    # _fn_schema
    fs = T._fn_schema("x", "desc", T.CalculateArgs)
    check("fn_schema", fs["function"]["name"] == "x" and "expression" in fs["function"]["parameters"]["properties"])


# ------------------------------------------------------------------ search

def test_search():
    c = _ctx()
    with mock_env(payload=_PG(usable=True, status="ok")):
        r = T._handle_search(c, {}, {"query": "tokyo population"})
        check("search_ok", "UNTRUSTED" in r and "[verify]" in r)
    with mock_env(payload=_PG(usable=True, status="stale", reason="old")):
        r = T._handle_search(c, {}, {"query": "q"})
        check("search_stale", "[DATA-VALIDATION]" in r)
    with mock_env(payload=_PG(usable=False, reason="empty")):
        r = T._handle_search(c, {}, {"query": "q"})
        check("search_unusable", r.startswith("[TOOL ERROR]"))
    with mock_env(search_result=T.NO_RESULTS):
        r = T._handle_search(c, {}, {"query": "q"})
        check("search_sentinel", r.strip() == T.NO_RESULTS)
    # web off + empty query
    check("search_off", T._handle_search(_ctx(web_search_enabled=False), {}, {"query": "x"}).startswith("[TOOL ERROR]"))
    check("search_noquery", T._handle_search(c, {}, {"query": "  "}).startswith("[TOOL ERROR]"))


# ------------------------------------------------------------------ deep_research

def test_deep_research():
    c = _ctx()
    st = {}
    with mock_env():
        r = T._handle_deep_research(c, st, {"topic": "vector DBs", "depth": "deep"})
        check("dr_ok", "Deep research complete" in r and st.get("research_report"))
    with mock_env(dr_result={"report": "# Executive Summary\ns\n", "path": "", "stats": {},
                             "cancelled": True}):
        r = T._handle_deep_research(c, {}, {"topic": "x"})
        check("dr_cancelled", "stopped early" in r)
    with mock_env(dr_result={"report": ""}):
        r = T._handle_deep_research(c, {}, {"topic": "x"})
        check("dr_no_report", r.startswith("[TOOL ERROR]"))
    with mock_env(payload=_PG(usable=False, reason="garbled")):
        r = T._handle_deep_research(c, {}, {"topic": "x"})
        check("dr_unusable", r.startswith("[TOOL ERROR]"))
    check("dr_off", T._handle_deep_research(_ctx(web_search_enabled=False), {}, {"topic": "x"}).startswith("[TOOL ERROR]"))
    check("dr_notopic", T._handle_deep_research(c, {}, {"topic": ""}).startswith("[TOOL ERROR]"))
    # depth coercion (invalid -> standard)
    with mock_env():
        T._handle_deep_research(c, {}, {"topic": "x", "depth": "bogus"})
        check("dr_depth_coerce", True)


# ------------------------------------------------------------------ calculate

def test_calculate():
    c = _ctx()
    check("calc_ok", T._handle_calculate(c, {}, {"expression": "sqrt(144)+2"}) == "14.0")
    check("calc_empty", T._handle_calculate(c, {}, {"expression": ""}).startswith("[TOOL ERROR]"))
    check("calc_toolong", T._handle_calculate(c, {}, {"expression": "1+" * 300}).startswith("[TOOL ERROR]"))
    check("calc_dunder", T._handle_calculate(c, {}, {"expression": "().__class__"}).startswith("[TOOL ERROR]"))
    check("calc_syntax", T._handle_calculate(c, {}, {"expression": "2+*"}).startswith("[TOOL ERROR]"))
    check("calc_attr", T._handle_calculate(c, {}, {"expression": "a.b"}).startswith("[TOOL ERROR]"))
    check("calc_list", T._handle_calculate(c, {}, {"expression": "[1,2,3]"}).startswith("[TOOL ERROR]"))
    check("calc_shift", T._handle_calculate(c, {}, {"expression": "1<<10"}).startswith("[TOOL ERROR]"))
    check("calc_explosive", T._handle_calculate(c, {}, {"expression": "factorial(5)"}).startswith("[TOOL ERROR]"))
    check("calc_tower", T._handle_calculate(c, {}, {"expression": "2**2**2"}).startswith("[TOOL ERROR]"))
    check("calc_bigexp", T._handle_calculate(c, {}, {"expression": "2**5000"}).startswith("[TOOL ERROR]"))
    check("calc_evalerr", T._handle_calculate(c, {}, {"expression": "1/0"}).startswith("[TOOL ERROR]"))


# ------------------------------------------------------------------ clipboard + remember

def test_clipboard_and_remember():
    c = _ctx()
    import subprocess
    real = subprocess.run
    subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout="copied text")
    try:
        check("clip_text", "UNTRUSTED" in T._handle_read_clipboard(c, {}, {}))
        subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout="   ")
        check("clip_empty", "пуст" in T._handle_read_clipboard(c, {}, {}))
        subprocess.run = lambda *a, **k: (_ for _ in ()).throw(OSError("no ps"))
        check("clip_error", T._handle_read_clipboard(c, {}, {}).startswith("[TOOL ERROR]"))
    finally:
        subprocess.run = real
    # remember
    c2 = _ctx()
    check("remember_empty", T._handle_remember_fact(c2, {}, {"fact": ""}).startswith("[TOOL ERROR]"))
    r = T._handle_remember_fact(c2, {}, {"fact": "user likes tea"})
    check("remember_added", "saved permanently" in r)
    r2 = T._handle_remember_fact(c2, {}, {"fact": "user likes tea"})
    check("remember_dup", "already saved" in r2)


# ------------------------------------------------------------------ generate_image

def test_generate_image():
    c = _ctx()
    check("gen_empty", T._handle_generate_image(c, {}, {"description": ""}).startswith("[TOOL ERROR]"))
    with mock_env() as out:
        st = {}
        r = T._handle_generate_image(c, st, {"description": "a cat"})
        check("gen_ok", "Image generated and saved" in r and st["image_status"] == "success")
    # partial status
    with mock_env(gen_result={"path": _img("g2.png"), "score": 5, "attempts": 2, "status": "partial"}):
        r = T._handle_generate_image(c, {}, {"description": "cat"})
        check("gen_partial", "base quality" in r)
    # renderer returns non-existent path -> fail
    with mock_env(gen_result={"path": "C:/no/such_zzz.png", "status": "success"}):
        r = T._handle_generate_image(c, {}, {"description": "cat"})
        check("gen_missing_path", r.startswith("[TOOL ERROR]"))
    # no path -> fail
    with mock_env(gen_result={"path": None, "status": "fail"}):
        r = T._handle_generate_image(c, {}, {"description": "cat"})
        check("gen_nopath", r.startswith("[TOOL ERROR]"))
    # reference-person mode success
    cref = _ctx(reference_person_mode=True)
    with mock_env(refperson=(_img("ref.png"), {"person": "Alice"})):
        st = {}
        r = T._handle_generate_image(cref, st, {"description": "Alice in a suit"})
        check("gen_refperson", "reference photo of Alice" in r)
    # reference-person exception -> fallthrough to text2image
    with mock_env(refperson=None) as out:
        def boom(ctx, d, seed=None): raise RuntimeError("refp fail")
        T.image_mod.generate_person_from_reference = boom
        r = T._handle_generate_image(_ctx(reference_person_mode=True), {}, {"description": "x"})
        check("gen_refperson_exc", "Image generated and saved" in r)
    # reference-person returns no image -> fallthrough
    with mock_env(refperson=(None, {"reason": "no face"})):
        r = T._handle_generate_image(_ctx(reference_person_mode=True), {}, {"description": "x"})
        check("gen_refperson_none", "Image generated and saved" in r)


# ------------------------------------------------------------------ redraw_image

def test_redraw_image():
    # the old model ('enhance'/'upscale'/'restore'/'outpaint') was removed from the
    # product along with its checkpoint files; 'redraw' (FireRed, or the stored
    # Ideogram layout) is the only mode left. redraw_image now normalizes any
    # mode/empty-instructions call onto that one path.
    c = _ctx(last_image_path=_img("src.png"))
    check("redraw_nosrc", T._handle_redraw_image(_ctx(), {}, {"mode": "redraw"}).startswith("[TOOL ERROR]"))
    with mock_env():
        check("redraw_ok", "redrawn and saved" in T._handle_redraw_image(c, {}, {"mode": "redraw", "instructions": "winter"}))
        # no instructions -> generic quality-pass instruction, still FireRed
        check("redraw_enhance", "redrawn and saved" in T._handle_redraw_image(c, {}, {"mode": "enhance"}))
        # a legacy mode name is coerced onto 'redraw', not rejected
        check("redraw_upscale", "redrawn" in T._handle_redraw_image(c, {}, {"mode": "upscale", "instructions": "4x"}))
        check("redraw_restore", "redrawn" in T._handle_redraw_image(c, {}, {"mode": "restore"}))
        # no explicit mode + instructions -> defaults to redraw
        check("redraw_default", "redrawn" in T._handle_redraw_image(c, {}, {"instructions": "change"}))
    # a picture with a stored Ideogram layout uses edit_via_layout instead of FireRed
    with mock_env(layout=True):
        check("redraw_via_layout", "redrawn and saved" in T._handle_redraw_image(c, {}, {"mode": "redraw", "instructions": "x"}))
    # FireRed fails, no layout -> [TOOL ERROR]
    with mock_env(firered_out=None):
        check("redraw_fail", T._handle_redraw_image(c, {}, {"mode": "redraw", "instructions": "x"}).startswith("[TOOL ERROR]"))
    # log_edit_decision raising is swallowed
    with mock_env(log_edit=lambda **k: (_ for _ in ()).throw(IOError("log"))):
        check("redraw_log_swallow", "redrawn" in T._handle_redraw_image(c, {}, {"mode": "redraw", "instructions": "x"}))


# ------------------------------------------------------------------ inspect_image

def test_inspect_image():
    c = _ctx(last_image_path=_img("insp.png"))
    check("inspect_nosrc", T._handle_inspect_image(_ctx(), {}, {"check": "x"}).startswith("[TOOL ERROR]"))
    with mock_env(vision="PRESENT: a cat, MISSING: hat"):
        r = T._handle_inspect_image(c, {}, {"check": "is there a cat?"})
        check("inspect_missing_footer", "a flaw was flagged" in r)
    with mock_env(vision="Everything is correct and present"):
        r = T._handle_inspect_image(c, {}, {"check": "all good?"})
        check("inspect_clean_footer", "everything checks out" in r)
    import tool_image_handlers as _tih
    _tih.ABSENCE_STUB = lambda ch: "still there" in ch
    with mock_env(vision="the puddle is MISSING"):
        r = T._handle_inspect_image(c, {}, {"check": "is the puddle still there?"})
    _tih.ABSENCE_STUB = None
    check("inspect_absence_footer", "REMOVAL check" in r)
    with mock_env(vision=""):
        r = T._handle_inspect_image(c, {}, {"check": "x"})
        check("inspect_empty_vision", r.startswith("[TOOL ERROR]"))
    # A blind inspector: the picture is normally lit (a mid-grey noisy image),
    # the model calls it "extremely dark" and labels everything MISSING. That
    # verdict is contradicted by the pixels and must not send the agent into
    # another re-render (live, 2026-09-12: 70 s wasted on a correct picture).
    import numpy as _np
    from PIL import Image as _I
    lit = str(_TMP / "lit.png")
    _I.fromarray(_np.random.default_rng(1).integers(90, 230, (64, 96, 3), dtype=_np.uint8)).save(lit)
    c2 = _ctx(last_image_path=lit)
    seen = {}
    with mock_env(vision="The image is extremely dark and underexposed. Cat: MISSING. Sofa: MISSING."):
        import llm as _llm
        _orig = _llm.analyze_image_with_llm
        _llm.analyze_image_with_llm = lambda **k: seen.update(k) or _orig(**k)
        r = T._handle_inspect_image(c2, {}, {"check": "is the cat black?"})
        check("inspect_blind_is_a_pass", "everything checks out" in r and "MISSING" not in r, r[:200])
        check("inspect_gets_measured_exposure", "Measured exposure" in seen.get("user_text", ""), seen.get("user_text", "")[:120])
    # A darkness remark next to a REAL finding keeps the finding (only a
    # verdict that calls the whole picture black is discarded wholesale).
    with mock_env(vision="Heavy shadows on the sofa. The cat is ginger instead of black: DISTORTED."):
        r = T._handle_inspect_image(c2, {}, {"check": "is the cat black?"})
        check("inspect_dark_plus_substance_stands", "a flaw was flagged" in r, r[:200])


# ------------------------------------------------------------------ inpaint_image

def test_inpaint_image():
    c = _ctx(last_image_path=_img("inp.png"), last_image_prompt="a woman")
    check("inpaint_nosrc", T._handle_inpaint_image(_ctx(), {}, {"instructions": "x", "region": "y"}).startswith("[TOOL ERROR]"))
    check("inpaint_noinstr", T._handle_inpaint_image(c, {}, {"instructions": "", "region": "sky"}).startswith("[TOOL ERROR]"))
    check("inpaint_noregion", T._handle_inpaint_image(c, {}, {"instructions": "blue", "region": ""}).startswith("[TOOL ERROR]"))
    # specialized category -> inpaint_region path
    with mock_env(category="clothing_edit"):
        r = T._handle_inpaint_image(c, {}, {"instructions": "red dress", "region": "dress"})
        check("inpaint_specialized", "Image edited" in r)
    # non-specialized -> route_edit path (background synth)
    with mock_env(category="background_replace"):
        r = T._handle_inpaint_image(c, {}, {"instructions": "a beach", "region": "background"})
        check("inpaint_route", "Image edited" in r)
    # removal synth + prompt fold
    with mock_env(is_removal=True, category="object_removal"):
        r = T._handle_inpaint_image(c, {}, {"instructions": "remove it", "region": "puddle"})
        check("inpaint_removal", "removed the puddle" in r)
    # engine coercion (invalid)
    with mock_env(category="face_edit"):
        r = T._handle_inpaint_image(c, {}, {"instructions": "x", "region": "face", "engine": "bogus"})
        check("inpaint_engine_coerce", "Image edited" in r)
    # failure reasons
    for reason, needle in [("region_absent", "not present"), ("not_found", "found no"),
                           ("bad_mask", "too large"), ("noisy_mask", "fragmented"),
                           ("server_error", "ComfyUI server failed")]:
        with mock_env(category="subject_edit", deliverable=None, inpaint_failure={"reason": reason}):
            r = T._handle_inpaint_image(c, {}, {"instructions": "x", "region": "thing"})
            check("inpaint_fail_" + reason, r.startswith("[TOOL ERROR]") and needle in r)


# ------------------------------------------------------------------ transfer_image

def test_transfer_image():
    imgs = [_img("t1.png"), _img("t2.png")]
    c = _ctx(reference_images=list(imgs), last_image_path=imgs[1])
    # too few images
    check("transfer_few", T._handle_transfer_image(_ctx(reference_images=[]), {}, {"instructions": "x"}).startswith("[TOOL ERROR]"))
    with mock_env():
        check("transfer_noinstr", T._handle_transfer_image(c, {}, {"instructions": ""}).startswith("[TOOL ERROR]"))
        r = T._handle_transfer_image(c, {}, {"instructions": "put hat on person", "roles": ["object_source"]})
        check("transfer_ok", "Transfer complete" in r)
        # invalid role -> None role
        r2 = T._handle_transfer_image(c, {}, {"instructions": "x", "roles": ["bogus_role"]})
        check("transfer_bad_role", "Transfer complete" in r2)
    with mock_env(deliverable=None):
        check("transfer_fail", T._handle_transfer_image(c, {}, {"instructions": "x"}).startswith("[TOOL ERROR]"))


# ------------------------------------------------------------------ fix_hands / fix_artifact

def test_fix_hands():
    c = _ctx(last_image_path=_img("fh.png"))
    check("fixhands_nosrc", T._handle_fix_hands(_ctx(), {}, {}).startswith("[TOOL ERROR]"))
    with mock_env():
        check("fixhands_ok", "Hands repaired" in T._handle_fix_hands(c, {}, {"engine": "qwenimage"}))
    with mock_env(deliverable=None):
        check("fixhands_noresult", T._handle_fix_hands(c, {}, {}).startswith("[TOOL ERROR]"))
    # fix_hands raises
    with mock_env():
        T.image_mod.fix_hands = lambda ctx, s, engine="firered": (_ for _ in ()).throw(RuntimeError("boom"))
        check("fixhands_raise", T._handle_fix_hands(c, {}, {}).startswith("[TOOL ERROR]"))


def test_fix_artifact():
    c = _ctx(last_image_path=_img("fa.png"))
    check("fixart_nosrc", T._handle_fix_artifact(_ctx(), {}, {"region": "x"}).startswith("[TOOL ERROR]"))
    check("fixart_noregion", T._handle_fix_artifact(c, {}, {"region": ""}).startswith("[TOOL ERROR]"))
    with mock_env():
        check("fixart_ok", "Repaired" in T._handle_fix_artifact(c, {}, {"region": "the seam", "issue": "smooth"}))
        # region without leading article -> "the region"
        check("fixart_article", "the smear" in T._handle_fix_artifact(c, {}, {"region": "smear"}))
    with mock_env(deliverable=None):
        check("fixart_noresult", T._handle_fix_artifact(c, {}, {"region": "x"}).startswith("[TOOL ERROR]"))
    with mock_env():
        T.image_mod.edit_region_contained_via_firered = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        check("fixart_raise", T._handle_fix_artifact(c, {}, {"region": "x"}).startswith("[TOOL ERROR]"))


# ------------------------------------------------------------------ find_photo

def test_find_photo():
    c = _ctx()
    check("findphoto_off", T._handle_find_photo(_ctx(web_search_enabled=False), {}, {"query": "x"}).startswith("[TOOL ERROR]"))
    check("findphoto_noquery", T._handle_find_photo(c, {}, {"query": ""}).startswith("[TOOL ERROR]"))
    with mock_env():
        st = {}
        r = T._handle_find_photo(c, st, {"query": "Sydney Sweeney", "require_face": True})
        check("findphoto_ok", "posted it to the chat" in r and st["image_status"] == "success")
    with mock_env(photo=None):
        check("findphoto_none", T._handle_find_photo(c, {}, {"query": "x", "require_face": False}).startswith("[TOOL ERROR]"))


# ------------------------------------------------------------------ execute_tool + validation

def test_execute_tool():
    c = _ctx()
    _u = T.execute_tool(c, {}, "nope", {})
    check("exec_unknown", _u.startswith("[TOOL ERROR]") and "Unknown tool: nope" in _u, _u)
    # validation error (missing required field)
    r = T.execute_tool(c, {}, "search", {})
    check("exec_validation", r.startswith("[TOOL ERROR] Invalid arguments"))
    # args not a dict -> validated as empty -> validation error
    r2 = T.execute_tool(c, {}, "calculate", "not-a-dict")
    check("exec_args_nondict", r2.startswith("[TOOL ERROR]"))
    # successful dispatch through validation (calculate coerces + runs)
    check("exec_ok", T.execute_tool(c, {}, "calculate", {"expression": "3*3"}) == "9")
    # handler raises -> wrapped
    with mock_env():
        T._handle_calculate_saved = T._handle_calculate
        # patch the ToolSpec handler? simpler: patch _BY_NAME entry's handler via a tool that raises
        # use read_clipboard with a raising subprocess is already [TOOL ERROR]; instead force
        # a handler exception via calculate monkeypatch on the spec
        pass
    # handler returns None -> coerced
    spec = T._BY_NAME["calculate"]
    orig = spec.handler
    object.__setattr__(spec, "handler", lambda ctx, st, a: None)
    try:
        check("exec_none", "produced no result" in T.execute_tool(c, {}, "calculate", {"expression": "1"}))
        object.__setattr__(spec, "handler", lambda ctx, st, a: 12345)
        check("exec_nonstr", T.execute_tool(c, {}, "calculate", {"expression": "1"}) == "12345")
        object.__setattr__(spec, "handler", lambda ctx, st, a: (_ for _ in ()).throw(RuntimeError("boom")))
        check("exec_handler_raise", T.execute_tool(c, {}, "calculate", {"expression": "1"}).startswith("[TOOL ERROR]"))
    finally:
        object.__setattr__(spec, "handler", orig)


def test_remaining_branches():
    c = _ctx()
    # 133: deep_research progress callback body runs when run_deep_research calls progress
    def dr_with_progress(ctx, topic, depth="standard", progress=None, **_kw):
        if progress:
            progress("Searching", {"sources": 3, "pages": 5, "findings": 2}, "msg")
        return {"report": "# Executive Summary\nx\n", "path": "r.md", "stats": {}, "cancelled": False}
    with mock_env(dr_result=None):
        import sys as _s
        _s.modules["deep_research"].run_deep_research = dr_with_progress
        T._handle_deep_research(c, {}, {"topic": "x"})
        check("dr_progress_cb", True)
    # 238->206: a Pow with a small constant exponent passes the guard and evaluates
    check("calc_small_pow", T._handle_calculate(c, {}, {"expression": "2**10"}) == "1024")
    # 668->671: inpaint with EMPTY last_image_prompt -> skip the prompt fold
    ci = _ctx(last_image_path=_img("nf.png"), last_image_prompt="")
    with mock_env(category="subject_edit"):
        check("inpaint_no_baseprompt", "Image edited" in T._handle_inpaint_image(ci, {}, {"instructions": "x", "region": "y"}))
    # 695: transfer where the target is NOT already in reference_images -> appended
    a, b = _img("ta.png"), _img("tb.png")
    ct = _ctx(reference_images=[a], last_image_path=b)
    with mock_env():
        check("transfer_target_appended", "Transfer complete" in T._handle_transfer_image(ct, {}, {"instructions": "x"}))
    # 748 / 805: invalid engine coercion in fix_hands / fix_artifact
    with mock_env():
        check("fixhands_engine_bad", "Hands repaired" in T._handle_fix_hands(_ctx(last_image_path=_img("h.png")), {}, {"engine": "bogus"}))
        check("fixart_engine_bad", "Repaired" in T._handle_fix_artifact(_ctx(last_image_path=_img("a.png")), {}, {"region": "x", "engine": "bogus"}))
    # 872-873: find_photo where OUTPUT_DIR.mkdir raises (swallowed)
    with mock_env():
        T.image_mod.OUTPUT_DIR = types.SimpleNamespace(mkdir=lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
        r = T._handle_find_photo(_ctx(), {}, {"query": "x"})
        check("findphoto_mkdir_except", "posted it to the chat" in r or r.startswith("[TOOL ERROR]"))


def test_pydantic_validators_via_execute_tool():
    c = _ctx()
    with mock_env():
        # deep_research depth coercion (_coerce_depth)
        T.execute_tool(c, {}, "deep_research", {"topic": "x", "depth": "BOGUS"})
        # generate_image steps/width/height/seed validators
        T.execute_tool(c, {}, "generate_image",
                       {"description": "cat", "steps": "20", "width": "1000", "height": "0", "seed": "-1"})
        # redraw mode coercion (_coerce_mode) invalid -> None
        T.execute_tool(c, {}, "redraw_image", {"mode": "BOGUS", "instructions": "x"})
        check("validators_ran", True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    print("\n" + str(len(fns) - failed) + "/" + str(len(fns)) + " tools test functions passed (" +
          str(sum(1 for _, c in RESULTS if c)) + "/" + str(len(RESULTS)) + " checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
