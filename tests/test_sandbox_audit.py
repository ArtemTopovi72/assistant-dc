"""Host-side sandbox helpers must check paths created by sandbox code too."""
import os
import subprocess
from types import SimpleNamespace

import pytest

from code_sandbox import Sandbox, SandboxError
import code_runner
import tool_code_handlers as handlers
import photo_search


def _link_dir(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True)
        if result.returncode:
            pytest.skip("directory links unavailable")


@pytest.fixture
def linked(tmp_path):
    box = Sandbox(tmp_path / "sandbox")
    outside = tmp_path / "outside"
    outside.mkdir()
    _link_dir(box.root / "linked", outside)
    return box, outside


def test_listing_does_not_reveal_linked_directory_contents(linked):
    box, outside = linked
    (outside / "private.txt").write_text("private")
    assert not any("private.txt" in row for row in box.list_dir(depth=3))


def test_outline_does_not_read_source_outside_sandbox(linked):
    box, outside = linked
    (outside / "private.py").write_text("def private_secret(): pass\n")
    result = handlers._handle_code_outline(SimpleNamespace(sandbox=box), {}, {})
    assert "private_secret" not in result


def test_run_code_does_not_deliver_pictures_outside_sandbox(linked):
    box, outside = linked
    (outside / "private.png").write_bytes(b"private")
    assert not handlers._pictures_in(box)


def test_private_agent_paths_cannot_write_outside_sandbox(tmp_path):
    box = Sandbox(tmp_path / "sandbox")
    outside = tmp_path / "outside"
    outside.mkdir()
    _link_dir(box.root / ".agent", outside)
    ctx = SimpleNamespace(deps=SimpleNamespace(
        docker_available=lambda **kw: False, desktop_binary=lambda: None))
    with pytest.raises(SandboxError, match="outside"):
        code_runner.run_python(box, "print('never run')", ctx=ctx)
    assert not list(outside.iterdir())


def test_private_tool_metadata_cannot_write_outside_sandbox(tmp_path):
    box = Sandbox(tmp_path / "sandbox")
    outside = tmp_path / "outside"
    outside.mkdir()
    _link_dir(box.root / ".agent", outside)
    ctx = SimpleNamespace(sandbox=box)
    handlers._record(box, "test", "private")
    handlers._remember_found(box, ["private"])
    result = handlers._handle_update_plan(ctx, {}, {"steps": ["private"]})
    assert result.startswith("[TOOL ERROR]"), result
    assert not list(outside.iterdir())


def test_private_tool_metadata_cannot_read_outside_sandbox(tmp_path):
    box = Sandbox(tmp_path / "sandbox")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "record.txt").write_text("private_secret")
    (outside / "plan.json").write_text('[{"step":"private_secret","status":"todo"}]')
    (outside / "found_photos.json").write_text('["private_secret"]')
    _link_dir(box.root / ".agent", outside)
    assert not handlers.read_record(box)
    assert not handlers._recall_found(box)


def test_photo_search_skips_pictures_outside_sandbox(linked):
    box, outside = linked
    (outside / "private.png").write_bytes(b"private")
    assert not photo_search.list_photos(box.root)


@pytest.mark.parametrize("remembered", [True, False])
def test_dedupe_does_not_copy_external_pictures(linked, remembered):
    from PIL import Image
    box, outside = linked
    secret = outside / "private.png"
    Image.new("RGB", (24, 24)).save(secret)
    ctx = SimpleNamespace(sandbox=box)
    if remembered:
        handlers._remember_found(box, [secret])
    result = handlers._handle_dedupe_photos(ctx, {}, {"keep_dir": "copies"})
    assert result.startswith("[TOOL ERROR]"), result
    assert not (box.root / "copies" / "private.png").exists()


def test_find_content_cannot_write_contact_sheets_outside_sandbox(tmp_path):
    from PIL import Image
    box = Sandbox(tmp_path / "sandbox")
    Image.new("RGB", (24, 24)).save(box.root / "picture.png")
    outside = tmp_path / "outside"
    outside.mkdir()
    _link_dir(box.root / ".contact_sheets", outside)
    ctx = SimpleNamespace(sandbox=box, vision_model=lambda *a, **kw: "[]")
    result = handlers._handle_find_content(ctx, {}, {"query": "a picture", "kind": "photos"})
    assert result.startswith("[TOOL ERROR]"), result
    assert not list(outside.iterdir())


def test_text_sniffing_reads_only_the_sample():
    import io
    reads = []

    class Stream(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    full = SimpleNamespace(suffix=".log", open=lambda mode: Stream(b"log text"),
                           read_bytes=lambda: (reads.append(-1) or b"log text"))
    assert Sandbox.is_texty(full)
    assert reads == [2048]


def test_undo_keeps_colliding_paths_separate(tmp_path):
    box = Sandbox(tmp_path)
    ctx = SimpleNamespace(sandbox=box)
    box.write_text("a/b.txt", "nested original")
    box.write_text("a__b.txt", "flat original")
    handlers._handle_write_file(ctx, {}, {"path": "a/b.txt", "content": "nested edit"})
    handlers._handle_write_file(ctx, {}, {"path": "a__b.txt", "content": "flat edit"})
    assert "restored" in handlers._handle_undo_edit(ctx, {}, {"path": "a/b.txt"})
    assert box.read_text("a/b.txt") == "nested original"
    assert "restored" in handlers._handle_undo_edit(ctx, {}, {"path": "a__b.txt"})
    assert box.read_text("a__b.txt") == "flat original"


@pytest.mark.parametrize("rel", ["a/b.txt", "a__b.txt"])
def test_undo_refuses_unattributable_legacy_history(tmp_path, rel):
    box = Sandbox(tmp_path)
    box.write_text(rel, "current")
    snapshot = box.resolve(".agent/undo/a__b.txt/1.bak")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("possibly another file")
    result = handlers._handle_undo_edit(SimpleNamespace(sandbox=box), {}, {"path": rel})
    assert result.startswith("[TOOL ERROR]") and "ambiguous" in result.lower(), result
    assert box.read_text(rel) == "current"
    assert snapshot.read_text() == "possibly another file"


def test_undo_can_still_restore_unambiguous_legacy_history(tmp_path):
    box = Sandbox(tmp_path)
    box.write_text("plain.txt", "current")
    snapshot = box.resolve(".agent/undo/plain.txt/1.bak")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("original")
    result = handlers._handle_undo_edit(SimpleNamespace(sandbox=box), {}, {"path": "plain.txt"})
    assert "restored" in result
    assert box.read_text("plain.txt") == "original"


@pytest.mark.parametrize("action", ["photos", "pictures", "outline", "search", "pack", "usage", "dedupe"])
def test_recursive_helpers_prune_external_directories_before_scanning(linked, action):
    import sys
    box, outside = linked
    (outside / "nested").mkdir()
    (outside / "nested" / "private.py").write_text("def private(): pass")
    box.write_text("inside.txt", "inside")
    script = '''
import sys
from pathlib import Path
from types import SimpleNamespace
from code_sandbox import Sandbox
import photo_search
import tool_code_handlers as handlers
box = Sandbox(sys.argv[1])
outside = Path(sys.argv[2]).resolve()
denied = []
def audit(event, args):
    if event == "os.scandir" and Path(args[0]).resolve().is_relative_to(outside):
        denied.append(str(args[0]))
        raise RuntimeError("external directory scan")
sys.addaudithook(audit)
action = sys.argv[3]
if action == "photos": photo_search.list_photos(box.root)
elif action == "pictures": handlers._pictures_in(box)
elif action == "outline": handlers._handle_code_outline(SimpleNamespace(sandbox=box), {}, {})
elif action == "search": box.search("inside")
elif action == "pack": box.pack(".", "packed.zip")
elif action == "usage": box.usage()
elif action == "dedupe": handlers._handle_dedupe_photos(SimpleNamespace(sandbox=box), {}, {})
assert not denied, denied
'''
    result = subprocess.run([sys.executable, "-c", script, str(box.root), str(outside), action],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_photo_search_accepts_relative_folder_paths(tmp_path):
    import sys
    from PIL import Image
    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (24, 24)).save(folder / "picture.png")
    script = '''
from pathlib import Path
import photo_search
assert len(photo_search.list_photos(Path("photos"))) == 1
result = photo_search.find_in_photos(None, Path("photos"), "a picture", vision=lambda *args: "[]")
assert result["scanned"] == 1
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
