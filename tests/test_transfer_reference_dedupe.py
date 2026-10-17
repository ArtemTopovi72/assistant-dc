"""transfer_image references never include a copy of the target itself."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import tool_image_handlers as H  # noqa: E402


def test_transfer_drops_a_byte_copy_of_the_target_from_references(tmp_path, monkeypatch):
    """The _working_input_* copy of the person photo came back as a
    clothing reference -> two passes, the first re-dressing her in her own sweater."""
    orig = tmp_path / "person.jpg"; orig.write_bytes(b"PERSON" * 100)
    work = tmp_path / "_working_input_1.jpg"; work.write_bytes(orig.read_bytes())
    hoodie = tmp_path / "hoodie.jpg"; hoodie.write_bytes(b"HOODIE" * 100)
    got = {}
    def fake(ctx, target, refs, instr):
        got["target"], got["refs"] = target, [r.path for r in refs]
        return None
    class C:
        deps = types.SimpleNamespace(plan_and_execute_transfer=fake)
        reference_images = [str(orig), str(hoodie)]
        last_image_path = str(work)
        def set_stage(self, *_): pass
    H._handle_transfer_image(C(), {"image_path": str(work)}, {"instructions": "dress her"})
    assert got["target"] == str(work) and got["refs"] == [str(hoodie)], got


def test_roles_given_for_every_album_photo_skip_the_target(tmp_path):
    """«одень мужика в доспехи как у девушки на второй фотке» came with roles
    [identity_reference, clothing_source] for one reference: photo 1 (the man) took
    the first role, and the man's face was transferred instead of the armour."""
    man = tmp_path / "man.jpg"; man.write_bytes(b"MAN" * 100)
    work = tmp_path / "_working_input_1.jpg"; work.write_bytes(man.read_bytes())
    girl = tmp_path / "girl.jpg"; girl.write_bytes(b"GIRL" * 100)
    got = {}
    def fake(ctx, target, refs, instr):
        got["roles"] = [r.role for r in refs]
        return None
    class C:
        deps = types.SimpleNamespace(plan_and_execute_transfer=fake)
        reference_images = [str(man), str(girl)]
        last_image_path = str(work)
        def set_stage(self, *_): pass
    H._handle_transfer_image(C(), {"image_path": str(work)},
                             {"instructions": "dress the man in her armour",
                              "roles": ["identity_reference", "clothing_source"]})
    assert got["roles"] == ["clothing_source"], got
