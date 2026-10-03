"""dedupe_photos: same-scene grouping the model can CALL.

Journey 33 (2026-09-14), four runs: with assistant_tools.dedupe named in the
run_code description, the schema and the result note, the model still wrote
its own exact-bytes hash and reported the re-shots gone. A dedicated tool is
the lever that works.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from PIL import Image, ImageFilter
from code_sandbox import Sandbox
import tool_code_handlers as H


class _Ctx:
    def __init__(self, box): self.sandbox = box


def _scene(seed, size=(320, 240)):
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8))
    return img.resize(size, Image.NEAREST)


@pytest.fixture
def box(tmp_path):
    b = Sandbox(tmp_path / "sb")
    d = b.root / "pics"; d.mkdir()
    a = _scene(1)
    a.save(d / "a_sharp.jpg", quality=95)
    # a re-shot: slightly shifted and blurred copy of the same scene
    a.transform(a.size, Image.AFFINE, (1, 0, 3, 0, 1, 2)).filter(ImageFilter.GaussianBlur(2)).save(d / "a_blur.jpg", quality=95)
    _scene(2).save(d / "b.jpg", quality=95)
    _scene(3).save(d / "c.png")
    return b


def test_groups_re_shots_and_keeps_the_sharpest(box):
    out = H._handle_dedupe_photos(_Ctx(box), {}, {"path": "pics"})
    assert out.startswith("4 photos -> 3 unique (1 near-duplicates dropped")
    assert "kept pics/a_sharp.jpg" in out
    kept = out.split("kept:")[1].split("group:")[0]
    assert "a_blur" not in kept and "b.jpg" in kept and "c.png" in kept


def test_explicit_paths_and_keep_dir(box):
    out = H._handle_dedupe_photos(_Ctx(box), {}, {
        "paths": ["pics/a_sharp.jpg", "pics/a_blur.jpg"], "keep_dir": "unique"})
    assert out.startswith("2 photos -> 1 unique")
    assert (box.root / "unique" / "a_sharp.jpg").exists()
    assert not (box.root / "unique" / "a_blur.jpg").exists()


def test_empty_folder_is_an_error(box):
    (box.root / "empty").mkdir()
    assert H._handle_dedupe_photos(_Ctx(box), {}, {"path": "empty"}).startswith("[TOOL ERROR]")


def test_missing_path_is_readable(box):
    out = H._handle_dedupe_photos(_Ctx(box), {}, {"paths": ["pics/nope.jpg"]})
    assert out.startswith("[TOOL ERROR]") and "nope.jpg" in out


def test_no_sandbox():
    class C: pass
    assert H._handle_dedupe_photos(C(), {}, {"path": "."}) == H._NO_SANDBOX


def test_registered_and_retrieved_on_a_dedupe_ask():
    import tools, tool_retrieval as T
    assert "dedupe_photos" in H.CODE_TOOL_NAMES
    assert any(s["function"]["name"] == "dedupe_photos" for s in tools.TOOL_SCHEMAS)


def test_hand_rolled_note_names_the_tool():
    note = H._hand_rolled_dedupe_note("import hashlib\nfor p in paths: h=hashlib.md5(open(p,'rb').read()).hexdigest() # dedupe duplicates")
    assert "dedupe_photos" in note


def test_defaults_to_what_find_content_found(box):
    ctx = _Ctx(box)
    ctx.found_image_paths = [str(box.root / "pics" / "a_sharp.jpg"), str(box.root / "pics" / "a_blur.jpg")]
    out = H._handle_dedupe_photos(ctx, {}, {})
    assert out.startswith("2 photos -> 1 unique") and "find_content found" in out
    assert "c.png" not in out


def test_whole_folder_needs_the_flag(box):
    ctx = _Ctx(box)
    ctx.found_image_paths = [str(box.root / "pics" / "a_sharp.jpg")]
    out = H._handle_dedupe_photos(ctx, {}, {"path": "pics"})
    assert out.startswith("1 photos -> 1 unique")          # found set still wins
    out = H._handle_dedupe_photos(ctx, {}, {"path": "pics", "whole_folder": True})
    assert out.startswith("4 photos -> 3 unique")


def test_find_content_result_survives_the_turn(box):
    # ctx is a per-turn copy in the bot; a fresh ctx must still see the set
    H._remember_found(box, [box.root / "pics" / "a_sharp.jpg", box.root / "pics" / "a_blur.jpg"])
    out = H._handle_dedupe_photos(_Ctx(box), {}, {})
    assert out.startswith("2 photos -> 1 unique") and "find_content found" in out


def test_photo_search_skips_hidden_folders(tmp_path):
    import photo_search as P
    (tmp_path / ".contact_sheets").mkdir(); (tmp_path / ".contact_sheets" / "sheet_000.jpg").write_bytes(b"x")
    (tmp_path / "d").mkdir(); (tmp_path / "d" / "p.jpg").write_bytes(b"x")
    assert [p.name for p in P.list_photos(tmp_path)] == ["p.jpg"]


def test_working_folder_record_survives_for_the_next_turn(tmp_path):
    """Torture run #4 2026-09-14: after compaction «сколько мостов нашёл?» got
    «я текстовый помощник, я не проводил поиск». The folder keeps a record and
    the message composer shows it."""
    import types
    import tool_code_handlers as H
    import graph_compose as C
    box = types.SimpleNamespace(root=tmp_path)
    assert H.read_record(box) == ""
    H._record(box, "find_content", "found 8 of 173 photos showing 'bridge': a.JPG, b.JPG")
    H._record(box, "dedupe_photos", "8 photos -> 4 kept")
    rec = H.read_record(box)
    assert "found 8 of 173" in rec and "4 kept" in rec
    ctx = types.SimpleNamespace(sandbox=box)
    assert "found 8 of 173" in C._working_folder_record(ctx)
    assert C._working_folder_record(types.SimpleNamespace()) == ""
    src = __import__("inspect").getsource(C._compose_user_message)
    assert "_working_folder_record(ctx)" in src and "never deny" in src
