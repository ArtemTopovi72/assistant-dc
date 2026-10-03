"""ComfyUI's input/output trees under runtime/ do not grow without bound.

- A delivered clip was COPIED out of runtime/comfy and the original kept, so
  every video was stored twice, forever.
- Reference images uploaded for each render piled up in runtime/comfy_in.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import utils  # noqa: E402
import video  # noqa: E402


def _age(p, days):
    t = time.time() - days * 86400
    os.utime(p, (t, t))


def test_old_uploads_are_removed_only_from_our_own_folder(tmp_path):
    runtime = tmp_path / "runtime"
    ours = runtime / "comfy_in"
    ours.mkdir(parents=True)
    old, new = ours / "ref_old.png", ours / "ref_new.png"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    _age(old, 30)
    assert utils.cleanup_comfy_inputs(ours, runtime) == 1
    assert not old.exists() and new.exists()
    # a shared ComfyUI input folder outside runtime/ is never touched
    shared = tmp_path / "ComfyUI" / "input"
    shared.mkdir(parents=True)
    mine = shared / "my_asset.png"
    mine.write_bytes(b"x")
    _age(mine, 365)
    assert utils.cleanup_comfy_inputs(shared, runtime) == 0
    assert mine.exists()


def test_adopted_clip_is_moved_not_copied(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    monkeypatch.setattr(video, "OUTPUT_DIR", str(out))
    src = tmp_path / "comfy" / "h3_00001_.mp4"
    src.parent.mkdir()
    src.write_bytes(b"clip")
    dest = video._adopt_output(str(src))
    assert open(dest, "rb").read() == b"clip"
    assert not src.exists()
    # a second clip with the same ComfyUI counter never overwrites the first
    src.write_bytes(b"clip2")
    dest2 = video._adopt_output(str(src))
    assert dest2 != dest and open(dest, "rb").read() == b"clip"


def test_failed_copy_keeps_the_original(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "OUTPUT_DIR", str(tmp_path / "file_not_dir"))
    (tmp_path / "file_not_dir").write_text("x")       # makedirs fails
    src = tmp_path / "c.mp4"
    src.write_bytes(b"clip")
    assert video._adopt_output(str(src)) == str(src) and src.exists()
