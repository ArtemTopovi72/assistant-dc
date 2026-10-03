"""The coding tools, end to end on the case that prompted them.

The scenario is the real one from the chat that started this work: someone
sends a .jar, and the assistant is supposed to open it, find the tag file that
governs the behaviour, add the missing block ids, pack it back up and send it —
instead of asking the user to paste source code at it.

The whole flow is exercised without an LLM. The model's job is choosing which
tool to call; that is measured on the routing bench. What is measured HERE is
that the tools do what their descriptions promise, and that a user without the
grant cannot reach them.
"""
import sys, os, json, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import tools as T
import sandbox_access as A
from code_sandbox import Sandbox


class _Ctx:
    def __init__(self, sandbox=None, user=None):
        self.sandbox = sandbox
        self.sandbox_user = user


class _User:
    def __init__(self, level):
        self.prefs = {"sandbox": level}


@pytest.fixture
def box(tmp_path):
    return Sandbox(tmp_path / "sbx")


@pytest.fixture
def ctx(box):
    return _Ctx(box, _User(A.CODE))


def _call(ctx, name, **args):
    return T.execute_tool(ctx, {}, name, args)


def _mod_jar(box, name="thief.jar"):
    with zipfile.ZipFile(box.root / name, "w") as zf:
        zf.writestr("data/thief/tags/block/break_protected/medium.json",
                    json.dumps({"values": ["#c:chests", "#c:villager_job_sites"]},
                               indent=2))
        zf.writestr("data/thief/tags/block/break_protected/heavy.json",
                    json.dumps({"values": []}))
        zf.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0")
    return name


# --- the scenario -----------------------------------------------------------

def test_the_whole_modpack_flow(ctx, box):
    _mod_jar(box)

    assert "thief.jar" in _call(ctx, "list_files", path=".")

    out = _call(ctx, "unpack_archive", path="thief.jar")
    assert "Unpacked" in out, out

    hits = _call(ctx, "search_files", pattern="villager_job_sites")
    assert "medium.json" in hits, hits

    shown = _call(ctx, "read_file",
                  path="thief_unpacked/data/thief/tags/block/break_protected/medium.json")
    assert "villager_job_sites" in shown
    assert "    1|" in shown, "read_file must number lines"

    edited = _call(ctx, "edit_file",
                   path="thief_unpacked/data/thief/tags/block/break_protected/medium.json",
                   old='"#c:villager_job_sites"',
                   new='"#c:villager_job_sites",\n    "morevillagers:trading_table"')
    assert "Edited" in edited, edited

    state = {}
    res = T.execute_tool(ctx, state, "pack_archive",
                         {"path": "thief_unpacked", "output": "thief_fixed.zip"})
    assert "Packed" in res, res

    # The artifact contract: nothing reaches the user unless this is set.
    assert state.get("document_path"), "the packed file was never handed over"
    with zipfile.ZipFile(state["document_path"]) as zf:
        data = json.loads(zf.read(
            "data/thief/tags/block/break_protected/medium.json"))
    assert "morevillagers:trading_table" in data["values"]


# --- the tools behave as advertised -----------------------------------------

def test_write_then_read_round_trip(ctx):
    _call(ctx, "write_file", path="pack.mcmeta", content='{"pack": {}}')
    assert '{"pack": {}}' in _call(ctx, "read_file", path="pack.mcmeta")


def test_an_ambiguous_edit_returns_a_correction_the_model_can_act_on(ctx):
    _call(ctx, "write_file", path="a.json", content='{"x": 1}\n{"x": 1}\n')
    out = _call(ctx, "edit_file", path="a.json", old='{"x": 1}', new='{"x": 2}')
    assert out.startswith("[TOOL ERROR]")
    assert "appears 2 times" in out, out


def test_a_path_outside_the_sandbox_is_a_tool_error_not_a_crash(ctx):
    out = _call(ctx, "read_file", path="../../../Windows/win.ini")
    assert out.startswith("[TOOL ERROR]") and "outside" in out


def test_searching_for_nothing_says_so_usefully(ctx):
    _call(ctx, "write_file", path="a.txt", content="hello")
    out = _call(ctx, "search_files", pattern="zzzz")
    assert "No line matches" in out


def test_missing_files_are_reported_plainly(ctx):
    assert "does not exist" in _call(ctx, "read_file", path="nope.json")


# --- access control ---------------------------------------------------------

def test_without_a_sandbox_the_tools_refuse(box):
    out = _call(_Ctx(None, _User(A.CODE)), "list_files", path=".")
    assert out.startswith("[TOOL ERROR]") and "No sandbox" in out


def test_the_files_level_cannot_run_code(box):
    ctx = _Ctx(box, _User(A.FILES))
    out = _call(ctx, "run_code", code="print(1)")
    assert out.startswith("[TOOL ERROR]")
    assert "not run code" in out or "may not install" in out


def test_the_files_level_cannot_install(box):
    out = _call(_Ctx(box, _User(A.FILES)), "install_packages", packages=["requests"])
    assert out.startswith("[TOOL ERROR]") and "may not install" in out


def test_a_user_with_no_level_at_all_cannot_run_code(box):
    out = _call(_Ctx(box, None), "run_code", code="print(1)")
    assert out.startswith("[TOOL ERROR]")


def test_code_execution_without_isolation_is_refused_not_silently_run(box, monkeypatch):
    """The rule from code_runner, checked at the tool boundary too."""
    import code_runner as R
    monkeypatch.setattr(R, "docker_available", lambda *a, **k: False)
    monkeypatch.setattr(R, "desktop_binary", lambda: None)
    out = _call(_Ctx(box, _User(A.CODE)), "run_code", code="print('hi')")
    assert out.startswith("[TOOL ERROR]") and "unavailable" in out.lower()


# --- what the model is offered ----------------------------------------------

def test_the_coding_schemas_stay_out_of_ordinary_turns():
    """Nine schemas riding along on every chat is how a context budget is lost."""
    import graph, tool_retrieval as TR
    from tool_code_handlers import CODE_TOOL_NAMES
    for text in ("нарисуй рыжего кота в скафандре",
                 "посчитай 15% от 2480",
                 "какая сейчас погода в Москве"):
        picked = {s["function"]["name"]
                  for s in TR.select_tools(text, graph.TOOL_SCHEMAS)}
        leaked = CODE_TOOL_NAMES & picked
        assert not leaked, f"{text!r} carried {sorted(leaked)}"


def test_the_file_tools_are_retrieved_as_a_kit():
    """Unpacking is useless without listing, reading and editing, and the model
    cannot ask for a schema it was never sent."""
    import graph, tool_retrieval as TR
    picked = {s["function"]["name"]
              for s in TR.select_tools("распакуй мод и добавь блоки в тег",
                                       graph.TOOL_SCHEMAS, wants=["unpack_archive"])}
    for needed in ("unpack_archive", "list_files", "read_file", "edit_file",
                   "pack_archive"):
        assert needed in picked, f"{needed} missing: {sorted(picked)}"


# --- a picture in the folder --------------------------------------------------

def _png(box, name="shot.png"):
    (box.root / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    return name


def test_a_picture_in_the_folder_can_be_opened_for_looking(ctx, box):
    """The requirement was "any input file -- code, an archive, a photo", and
    photos were the hole: read_file refuses a PNG, and inspect_image only ever
    looked at state["image_path"], which a file in the working folder never
    reached. A user could send a screenshot, see it listed, and be told it
    could not be read."""
    _png(box)
    state = {}
    out = T.execute_tool(ctx, state, "open_image", {"path": "shot.png"})
    assert not out.startswith("[TOOL ERROR]"), out
    assert state.get("image_path", "").endswith("shot.png"), state
    assert "inspect_image" in out, out


def test_open_image_refuses_a_text_file_with_a_usable_correction(ctx):
    _call(ctx, "write_file", path="notes.txt", content="hi")
    out = _call(ctx, "open_image", path="notes.txt")
    assert out.startswith("[TOOL ERROR]") and "read_file" in out, out


def test_open_image_cannot_reach_outside_the_sandbox(ctx):
    out = _call(ctx, "open_image", path="../../secret.png")
    assert out.startswith("[TOOL ERROR]") and "outside" in out, out


def test_open_image_is_offered_with_the_rest_of_the_file_kit():
    """The model cannot call a schema it was never sent -- it would tell the
    user to re-send a photo that is already sitting in their folder."""
    import graph, tool_retrieval as TR
    # the model's read of each (agent/intent.py `wants`)
    for text, wants in (("распакуй мод и добавь блоки в тег", ["unpack_archive"]),
                        ("посмотри что на скриншоте", ["open_image"])):
        picked = {s["function"]["name"]
                  for s in TR.select_tools(text, graph.TOOL_SCHEMAS, wants=wants)}
        assert "open_image" in picked, (text, sorted(picked))
        assert "inspect_image" in picked, (text, sorted(picked))


# --- being unable to look must not become inventing ---------------------------

def test_a_named_picture_in_the_folder_forces_a_look(box):
    """Measured with open_image withheld: asked what was in scan.png, the model
    did not refuse and did not ask -- it answered "нарисован синий круг" about a
    red triangle. Given a file it cannot read, it invents a description. So
    looking is made the only thing it CAN do on that round."""
    import graph_personality as P
    _png(box, "scan.png")
    offered = [{"function": {"name": "open_image"}}]
    assert P._forced_look("посмотри что на scan.png", offered, set(), box) == "open_image"


def test_the_force_is_narrow(box):
    import graph_personality as P
    _png(box, "scan.png")
    offered = [{"function": {"name": "open_image"}}]
    # already looked
    assert not P._forced_look("scan.png", offered, {"open_image"}, box)
    assert not P._forced_look("scan.png", offered, {"inspect_image"}, box)
    # a file that is not there
    assert not P._forced_look("other.png", offered, set(), box)
    # not an image
    assert not P._forced_look("config.json", offered, set(), box)
    # the tool was not offered this round
    assert not P._forced_look("scan.png", [{"function": {"name": "read_file"}}],
                              set(), box)
    # no sandbox at all
    assert not P._forced_look("scan.png", offered, set(), None)


def test_a_hostile_filename_cannot_escape_while_being_looked_for(box):
    """The text is user-controlled, and this reaches the filesystem."""
    import graph_personality as P
    offered = [{"function": {"name": "open_image"}}]
    for hostile in ("../../../Windows/win.png", "C:/Windows/win.png",
                    "//server/share/x.png"):
        assert not P._forced_look(hostile, offered, set(), box), hostile


# --- the loop actually forces the look ---------------------------------------
# Not a source grep: a grep of the call site has twice passed against mutations
# that broke the behaviour. This drives the real loop and reads the tool_choice
# that reached the LLM boundary.

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_tgf2", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_graph_full.py"))
_TGF = _ilu.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_TGF)
except SystemExit:
    pass                      # that module runs its own checks at import


def _rounds_with_sandbox(text, box, user_level=A.CODE):
    llm = _TGF.LLM([
        _TGF._asst("", [_TGF._tc("open_image", {"path": "scan.png"}, "o1")]),
        _TGF._asst("Красный треугольник."),
    ])
    tools = _TGF.Tools({"open_image": "Opened 'scan.png'."})
    ctx = _TGF._ctx()
    ctx.sandbox = box
    ctx.sandbox_user = _User(user_level)
    _TGF._run(ctx, llm, tools, {"user_input": text, "messages": []},
              fastpath=False)
    return [(c["kw"].get("tool_choice"),
             [t.get("function", {}).get("name") for t in (c["kw"].get("tools") or [])])
            for c in llm.calls]


def test_the_loop_forces_open_image_for_a_picture_in_the_folder(box):
    _png(box, "scan.png")
    rounds = _rounds_with_sandbox("посмотри что на scan.png", box)
    assert rounds, "the loop made no LLM call"
    choice, names = rounds[0]
    # Forcing = only that one schema offered. It goes out as "auto" first
    # (graph_personality: "required" is grammar-constrained and made Gemma
    # reason for minutes); "required" is the retry when it answers in text.
    assert choice in ("auto", "required"), rounds
    assert names == ["open_image"], names


def test_the_force_is_released_once_it_has_looked(box):
    """Round two must be free, or it can never say what it saw."""
    _png(box, "scan.png")
    rounds = _rounds_with_sandbox("посмотри что на scan.png", box)
    assert len(rounds) > 1 and rounds[1][0] == "auto", rounds


def test_no_force_when_the_file_is_not_there(box):
    rounds = _rounds_with_sandbox("посмотри что на scan.png", box)
    assert rounds[0][0] == "auto", rounds


# --- files in the folder outrank the cue vocabulary --------------------------

def test_a_folder_with_files_keeps_the_file_kit_on_the_turn(box):
    """Measured: "тут два джарника, посмотри в обоих" retrieved ZERO file tools
    -- the cue table knows "распакуй" and "архив" but not "джарник" -- so the
    model had nothing to open them with and asked the user to paste their
    contents, which is the exact complaint this feature was built to fix.

    A vocabulary can always be missing a word. The folder either has files or it
    does not.
    """
    import graph_personality as P
    assert not P._sandbox_has_files(None)
    assert not P._sandbox_has_files(box)
    _mod_jar(box, "thief.jar")
    assert P._sandbox_has_files(box)


def test_the_kit_is_pinned_exactly_when_the_folder_has_files(box):
    """The decision itself, not the loop's source and not a stubbed harness --
    a harness that replaces the schema list cannot tell this apart from
    retrieval's send-everything fallback, so it proved nothing."""
    import graph_personality as P, tool_retrieval as TR

    class _C:
        sandbox = None

    c = _C()
    assert P._kit_always(c, {"search"}) == {"search"}
    c.sandbox = box
    assert P._kit_always(c, set()) == set(), "an empty folder pins nothing"
    _mod_jar(box, "thief.jar")
    pinned = P._kit_always(c, {"search"})
    assert "search" in pinned, pinned
    for needed in ("list_files", "read_file", "unpack_archive"):
        assert needed in pinned, (needed, sorted(pinned))


def test_an_empty_folder_does_not_pin_the_kit(box):
    """The narrowing exists to fit the context budget, so the rule has to be
    "the folder has files", not "the folder exists".

    Driven through select_tools with the two `always` sets the loop builds, so
    the assertion is about the decision and not about a prompt that happens to
    match no cue at all (that case sends everything by design -- no evidence is
    not evidence of absence)."""
    import graph, tool_retrieval as TR
    from tool_code_handlers import CODE_TOOL_NAMES
    text = "посчитай 15% от 2480"

    empty = {s["function"]["name"]
             for s in TR.select_tools(text, graph.TOOL_SCHEMAS, always=set())}
    assert not (CODE_TOOL_NAMES & empty), sorted(empty)

    full = {s["function"]["name"]
            for s in TR.select_tools(text, graph.TOOL_SCHEMAS,
                                     always=set(TR._CODE_KIT))}
    assert "unpack_archive" in full, sorted(full)


# --- a delivered archive is opened, not narrated about ------------------------

def test_a_delivered_archive_forces_unpack(box):
    """Live 2026-09-14: a 112 MB DCIM.zip reached the sandbox with the
    "[The file ... is now in your working folder]" note; the model wrote
    «Сначала я распакую архив ... Начинаю поиск.» and ended the turn with no
    tool call. Every consumer idled; the user waited on a promise."""
    import graph_personality as P
    import zipfile
    with zipfile.ZipFile(box.root / "DCIM.zip", "w") as z:
        z.writestr("a.txt", "x")
    hint = "найди мост\n[The file 'DCIM.zip' is now in your working folder. Open it yourself with: list_files, unpack_archive, read_file.]"
    offered = [{"function": {"name": "unpack_archive"}}, {"function": {"name": "list_files"}}]
    assert P._forced_unpack(hint, offered, set(), box) == "unpack_archive"
    # already opened this turn: the model is writing its answer
    assert not P._forced_unpack(hint, offered, {"unpack_archive"}, box)
    assert not P._forced_unpack(hint, offered, {"list_files"}, box)
    # not an archive, or not on offer, or not actually there
    assert not P._forced_unpack("[The file 'notes.txt' is now in your working folder.", offered, set(), box)
    assert not P._forced_unpack(hint, [{"function": {"name": "read_file"}}], set(), box)
    assert not P._forced_unpack("[The file 'gone.zip' is now in your working folder.", offered, set(), box)
    # the rule is wired into the round's forcing chain
    import inspect
    src = inspect.getsource(P.personality_node)
    assert "_forced_unpack(" in src


# --- no "done" on code nobody ran (mini-swe-agent's loop) --------------------

def _edit_then_answer(box, script, user_level=A.CODE):
    llm = _TGF.LLM(script)
    tools = _TGF.Tools({"edit_file": "Edited build_mod.py.",
                        "run_code": "exit 0\nstdout: ok"})
    ctx = _TGF._ctx()
    ctx.sandbox = box
    ctx.sandbox_user = _User(user_level)
    out = _TGF._run(ctx, llm, tools, {"user_input": "почини build_mod.py", "messages": []},
                    fastpath=False)
    return llm, tools, out


_EDIT = {"path": "build_mod.py", "old": "print(1)", "new": "print(2)"}


def test_an_answer_after_an_unrun_edit_is_sent_back_to_run_it(box):
    box.write_text("build_mod.py", "print(1)\n")
    llm, tools, out = _edit_then_answer(box, [
        _TGF._asst("", [_TGF._tc("edit_file", _EDIT, "e1")]),
        _TGF._asst("Готово, исправил."),                       # vouches, never ran it
        _TGF._asst("", [_TGF._tc("run_code", {"code": "import build_mod"}, "r1")]),
        _TGF._asst("Запустил: печатает 2, работает."),
    ])
    names = [n for n, _ in tools.calls]
    assert names == ["edit_file", "run_code"], names
    nudge = " ".join(str(m.get("content")) for m in llm.calls[2]["messages"])
    assert "have not run it since" in nudge, nudge[-400:]
    assert "Запустил" in (out.get("final_answer") or ""), out.get("final_answer")


def test_edit_then_run_then_answer_passes_untouched(box):
    box.write_text("build_mod.py", "print(1)\n")
    llm, tools, out = _edit_then_answer(box, [
        _TGF._asst("", [_TGF._tc("edit_file", _EDIT, "e1")]),
        _TGF._asst("", [_TGF._tc("run_code", {"code": "import build_mod"}, "r1")]),
        _TGF._asst("Исправил и запустил — работает."),
    ])
    assert len(llm.calls) == 3, len(llm.calls)
    assert "работает" in (out.get("final_answer") or "")


def test_no_run_is_demanded_from_an_account_that_cannot_run_code(box):
    box.write_text("build_mod.py", "print(1)\n")
    llm, tools, out = _edit_then_answer(box, [
        _TGF._asst("", [_TGF._tc("edit_file", _EDIT, "e1")]),
        _TGF._asst("Поправил файл."),
    ], user_level=A.FILES)
    assert len(llm.calls) == 2, len(llm.calls)
    assert "Поправил" in (out.get("final_answer") or "")


# --- changed-but-never-packed guard (real-model bench 2026-09-23) ------------

def _unpack_then(box, script, *, touch=True):
    import time as _t
    (box.root / "mod_unpacked").mkdir()
    f = box.root / "mod_unpacked" / "tag.json"
    f.write_text("{}")
    _old = _t.time() - 60
    os.utime(f, (_old, _old))

    def _run_code(args):
        if touch:
            _t.sleep(0.05)          # a real edit comes seconds after the unpack
            f.write_text('{"values": ["x"]}')
        return "exit=0 in 0.1s"
    llm = _TGF.LLM(script)
    tools = _TGF.Tools({"unpack_archive": "Unpacked into mod_unpacked. List it to see what is inside.",
                        "run_code": _run_code,
                        "pack_archive": "Packed mod_unpacked into mod.jar."})
    ctx = _TGF._ctx()
    ctx.sandbox = box
    ctx.sandbox_user = _User(A.CODE)
    out = _TGF._run(ctx, llm, tools, {"user_input": "почини мод", "messages": []},
                    fastpath=False)
    return llm, tools, out


def test_changed_unpacked_folder_without_pack_is_sent_back(box):
    llm, tools, out = _unpack_then(box, [
        _TGF._asst("", [_TGF._tc("unpack_archive", {"path": "mod.jar"}, "u1")]),
        _TGF._asst("", [_TGF._tc("run_code", {"code": "edit"}, "r1")]),
        _TGF._asst("Я добавил блоки в тег, теперь работает."),     # never packed
        _TGF._asst("", [_TGF._tc("pack_archive", {"path": "mod_unpacked"}, "p1")]),
        _TGF._asst("Запаковал и отправил mod.jar."),
    ])
    names = [n for n, _ in tools.calls]
    assert names[-1] == "pack_archive", names
    nudge = " ".join(str(m.get("content")) for m in llm.calls[3]["messages"])
    assert "never packed" in nudge, nudge[-300:]


def test_reading_an_unpacked_archive_needs_no_pack(box):
    llm, tools, out = _unpack_then(box, [
        _TGF._asst("", [_TGF._tc("unpack_archive", {"path": "mod.jar"}, "u1")]),
        _TGF._asst("", [_TGF._tc("run_code", {"code": "read"}, "r1")]),
        _TGF._asst("В теге три блока."),
    ], touch=False)
    assert len(llm.calls) == 3, len(llm.calls)


def test_packing_the_other_folder_does_not_settle_the_changed_one(box):
    # Real-model run: edited MoreVillagers, packed Thief, answered.
    import time as _t
    for d in ("a_unpacked", "b_unpacked"):
        (box.root / d).mkdir()
        (box.root / d / "t.json").write_text("{}")
        os.utime(box.root / d / "t.json", (_t.time() - 60, _t.time() - 60))
    results = iter(["Unpacked into a_unpacked. List it.", "Unpacked into b_unpacked. List it."])

    def _edit_a(args):
        _t.sleep(0.05)
        (box.root / "a_unpacked" / "t.json").write_text('{"values": [1]}')
        return "exit=0 in 0.1s"
    llm = _TGF.LLM([
        _TGF._asst("", [_TGF._tc("unpack_archive", {"path": "a.jar"}, "u1")]),
        _TGF._asst("", [_TGF._tc("unpack_archive", {"path": "b.jar"}, "u2")]),
        _TGF._asst("", [_TGF._tc("run_code", {"code": "edit a"}, "r1")]),
        _TGF._asst("", [_TGF._tc("pack_archive", {"path": "b_unpacked"}, "p1")]),
        _TGF._asst("Готово, отправил архив."),
        _TGF._asst("", [_TGF._tc("pack_archive", {"path": "a_unpacked"}, "p2")]),
        _TGF._asst("Запаковал a."),
    ])
    tools = _TGF.Tools({"unpack_archive": lambda a: next(results),
                        "run_code": _edit_a, "pack_archive": "Packed."})
    ctx = _TGF._ctx(); ctx.sandbox = box; ctx.sandbox_user = _User(A.CODE)
    _TGF._run(ctx, llm, tools, {"user_input": "почини", "messages": []}, fastpath=False)
    names = [(n, a.get("path")) for n, a in tools.calls]
    assert names[-1] == ("pack_archive", "a_unpacked"), names


def test_a_repeated_write_says_nothing_changed(ctx):
    # Bench deliver_symptom 2026-09-23: five identical write_file calls, each
    # "Wrote ...", until the rounds ran out and the fix was never packed.
    _call(ctx, "write_file", path="a.json", content='{"x": 1}')
    again = _call(ctx, "write_file", path="a.json", content='{"x": 1}')
    assert "Nothing changed" in again and "pack_archive" in again


def test_a_repeated_edit_says_nothing_changed(ctx):
    _call(ctx, "write_file", path="b.txt", content="one\ntwo\n")
    first = _call(ctx, "edit_file", path="b.txt", old="two", new="two")
    assert "Nothing changed" in first


def test_a_turn_that_ends_with_unshipped_changes_is_packed(ctx, box):
    import time
    import graph_personality as GP
    _mod_jar(box)
    _call(ctx, "unpack_archive", path="thief.jar")
    since = time.time() - 1
    root = box.resolve("thief_unpacked")
    victim = next(f for f in root.rglob("*.json"))
    victim.write_text('{"values": ["morevillagers:trading_table"]}', encoding="utf-8")
    state = {}
    out = GP._auto_pack_leftover(ctx, state, {"thief_unpacked": since}, "Готово.")
    assert state.get("document_status") == "success"
    assert state["document_path"].endswith("thief_fixed.jar")
    assert zipfile.is_zipfile(state["document_path"])
    assert "thief_fixed.jar" in out


def test_nothing_is_packed_when_nothing_changed(ctx, box):
    import time
    import graph_personality as GP
    _mod_jar(box)
    _call(ctx, "unpack_archive", path="thief.jar")
    state = {}
    out = GP._auto_pack_leftover(ctx, state, {"thief_unpacked": time.time() + 5}, "Ответ.")
    assert out == "Ответ." and "document_path" not in state


def test_the_unpack_result_names_a_folder_the_guard_can_resolve(ctx, box):
    import re
    _mod_jar(box)
    out = _call(ctx, "unpack_archive", path="thief.jar")
    m = re.search(r"Unpacked into (.+?)(?: \(\d+ files?\))?\. ", out)
    assert m and box.resolve(m.group(1)).is_dir(), out


def test_packing_the_work_root_over_unpacked_mods_is_refused(ctx, box):
    _mod_jar(box)
    _call(ctx, "unpack_archive", path="thief.jar")
    res = _call(ctx, "pack_archive", path=".", output="all.zip")
    assert "Not packed" in res and "thief_unpacked" in res


def test_auto_pack_ships_the_folder_changed_last(ctx, box):
    import time
    import graph_personality as GP
    _mod_jar(box, "a.jar"); _mod_jar(box, "b.jar")
    _call(ctx, "unpack_archive", path="a.jar"); _call(ctx, "unpack_archive", path="b.jar")
    # The loop stamps `since` AFTER the unpack (graph_personality), so only
    # the edits below are newer than it, not the extracted files themselves.
    since = time.time() + 1
    for name, ahead in (("a_unpacked", 2), ("b_unpacked", 4)):
        f = next(box.resolve(name).rglob("medium.json"))
        f.write_text("{}", encoding="utf-8")
        t = time.time() + ahead
        os.utime(f, (t, t))
    state = {}
    GP._auto_pack_leftover(ctx, state, {"a_unpacked": since, "b_unpacked": since}, "ok")
    assert state["document_path"].endswith("b_fixed.jar")


def test_a_full_folder_pins_only_the_look_kit_until_it_is_used(ctx, box):
    import graph_personality as GP
    import tool_retrieval as R
    box.write_text("a.txt", "x")
    assert GP._kit_always(ctx, set()) == set(R._LOOK_KIT)
    assert set(R._CODE_KIT) <= GP._kit_always(ctx, {"read_file"})
    schemas = [{"function": {"name": n}} for n in R._CODE_KIT + ("search", "run_code")]
    picked = {s["function"]["name"] for s in R.select_tools("какая погода", schemas,
                                                            always=set(R._LOOK_KIT))}
    assert "edit_file" not in picked and "list_files" in picked
