"""transfer_image references never include a copy of the target itself."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import tool_image_handlers as H  # noqa: E402


def test_transfer_drops_a_byte_copy_of_the_target_from_references(tmp_path, monkeypatch):
    """Live 10-06: the _working_input_* copy of the person photo came back as a
    clothing reference -> two passes, the first re-dressing her in her own sweater."""
    import image as image_mod
    orig = tmp_path / "person.jpg"; orig.write_bytes(b"PERSON" * 100)
    work = tmp_path / "_working_input_1.jpg"; work.write_bytes(orig.read_bytes())
    hoodie = tmp_path / "hoodie.jpg"; hoodie.write_bytes(b"HOODIE" * 100)
    got = {}
    def fake(ctx, target, refs, instr):
        got["target"], got["refs"] = target, [r.path for r in refs]
        return None
    monkeypatch.setattr(image_mod, "plan_and_execute_transfer", fake)
    class C:
        reference_images = [str(orig), str(hoodie)]
        last_image_path = str(work)
        def set_stage(self, *_): pass
    H._handle_transfer_image(C(), {"image_path": str(work)}, {"instructions": "dress her"})
    assert got["target"] == str(work) and got["refs"] == [str(hoodie)], got
