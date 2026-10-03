"""delete_path: a contained, selective delete for the working folder.

Live 2026-09-14: «не захламляй: удали архив и папку с фотками, коллаж
оставь». There was no delete tool -- only run_code, which a FILES-level user
does not have -- so the clean-up the working folder invites was impossible.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from pathlib import Path
from code_sandbox import Sandbox, SandboxError
import tool_code_handlers as H


class _Ctx:
    def __init__(self, box): self.sandbox = box


@pytest.fixture
def box(tmp_path):
    b = Sandbox(tmp_path / "sb")
    (b.root / "DCIM.zip").write_bytes(b"PK")
    (b.root / "DCIM_unpacked" / "a").mkdir(parents=True)
    (b.root / "DCIM_unpacked" / "a" / "1.jpg").write_bytes(b"x")
    (b.root / "collage.jpg").write_bytes(b"y")
    (b.root / "make_collage.py").write_text("print(1)")
    return b


def test_deletes_a_file(box):
    out = H._handle_delete_path(_Ctx(box), {}, {"path": "DCIM.zip"})
    assert "Deleted file DCIM.zip" in out and not (box.root / "DCIM.zip").exists()


def test_deletes_a_folder_and_counts(box):
    out = H._handle_delete_path(_Ctx(box), {}, {"path": "DCIM_unpacked"})
    assert "folder DCIM_unpacked (1 files)" in out
    assert not (box.root / "DCIM_unpacked").exists()
    assert (box.root / "collage.jpg").exists() and (box.root / "make_collage.py").exists()


def test_the_root_empties_the_folder(box):
    # Live 2026-09-14: «удали все файлы» was refused here and the model
    # claimed it deleted everything anyway. Now it really does.
    out = H._handle_delete_path(_Ctx(box), {}, {"path": "."})
    assert "everything in the working folder" in out and "4 entries" in out
    assert not any(box.root.iterdir()) and box.root.exists()


def test_all_and_star_mean_everything(box):
    for rel in ("*", "all", ""):
        b = Sandbox(box.root.parent / f"sb_{len(rel)}")
        (b.root / "x.txt").write_text("x")
        assert "everything" in H._handle_delete_path(_Ctx(b), {}, {"path": rel})
        assert not any(b.root.iterdir())


def test_refuses_to_escape(box):
    out = H._handle_delete_path(_Ctx(box), {}, {"path": "../../etc"})
    assert out.startswith("[TOOL ERROR]")


def test_missing_path_is_a_readable_error(box):
    out = H._handle_delete_path(_Ctx(box), {}, {"path": "DCIM.zpi"})
    assert out.startswith("[TOOL ERROR]") and "does not exist" in out


def test_a_deleted_current_picture_is_forgotten(box):
    state = {"image_path": str(box.root / "collage.jpg"), "image_status": "ok"}
    H._handle_delete_path(_Ctx(box), state, {"path": "collage.jpg"})
    assert "image_path" not in state


def test_a_surviving_picture_stays_current(box):
    state = {"image_path": str(box.root / "collage.jpg")}
    H._handle_delete_path(_Ctx(box), state, {"path": "DCIM.zip"})
    assert state["image_path"].endswith("collage.jpg")


def test_no_sandbox(box):
    class _NoBox: sandbox = None
    assert H._handle_delete_path(_NoBox(), {}, {"path": "x"}).startswith("[TOOL ERROR]")


def test_it_is_in_the_kit_and_registered():
    import tool_retrieval as T, tools
    assert "delete_path" in T._CODE_KIT
    assert "delete_path" in H.CODE_TOOL_NAMES
    assert any(s["function"]["name"] == "delete_path" for s in tools.TOOL_SCHEMAS)


# --- a script that writes a picture delivers it (journey 32) ----------------

def _run_code_with(box, monkeypatch, writer):
    import code_runner as R, sandbox_access as A
    from tg_userstore import _User
    class _Res:
        ok = True; code = 0; output = "done"; seconds = 0.1; backend = "fake"
        def as_tool_result(self): return "exit=0 in 0.1s\ndone"
    def fake_run(sb, code, timeout=0, allow_host=False):
        writer(); return _Res()
    monkeypatch.setattr(R, "run_python", fake_run)
    user = _User(chat_id=1, name="u", status="approved"); user.prefs["sandbox"] = A.CODE
    class _C: sandbox = box; sandbox_user = user
    state = {"image_path": str(box.root / "DCIM_unpacked" / "a" / "1.jpg")}
    out = H._handle_run_code(_C(), state, {"code": "x"})
    return out, state


def test_a_picture_written_by_the_script_becomes_current(box, monkeypatch):
    def w(): (box.root / "out_collage.png").write_bytes(b"png")
    out, state = _run_code_with(box, monkeypatch, w)
    assert state["image_path"].endswith("out_collage.png") and state["image_status"] == "ok"
    assert "will be sent to the user" in out


def test_untouched_pictures_do_not_steal_the_current_one(box, monkeypatch):
    out, state = _run_code_with(box, monkeypatch, lambda: None)
    assert state["image_path"].endswith("1.jpg") and "image_status" not in state


def test_hidden_folders_are_not_deliverables(box, monkeypatch):
    def w():
        (box.root / ".contact_sheets").mkdir(); (box.root / ".contact_sheets" / "s.jpg").write_bytes(b"x")
    out, state = _run_code_with(box, monkeypatch, w)
    assert state["image_path"].endswith("1.jpg")


# --- a hand-rolled photo hash gets called out (journey 33) ------------------

def test_hand_rolled_dedupe_is_flagged():
    code = "h = hash(img.tobytes())\nif h in seen: print('duplicate')"
    assert "dedupe_photos" in H._hand_rolled_dedupe_note(code)

def test_using_the_helper_is_not_flagged():
    assert H._hand_rolled_dedupe_note("import assistant_tools\nk = assistant_tools.dedupe(ps)") == ""

def test_a_hash_with_no_dedupe_intent_is_not_flagged():
    assert H._hand_rolled_dedupe_note("import hashlib\nprint(hashlib.md5(b'x').hexdigest())") == ""


def test_a_failed_delete_followed_by_a_success_claim_is_caught():
    import graph_personality as G
    assert "delete_path" in G._FABRICATION_RISK and "run_code" in G._FABRICATION_RISK
    assert G._answer_contradicts_reality("Я удалил все файлы из рабочей директории.",
                                         False, {"delete_path"}, False)
    assert not G._answer_contradicts_reality("Не получилось: инструмент отказал, папка не тронута.",
                                             False, {"delete_path"}, False)
