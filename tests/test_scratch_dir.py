"""Intermediate renders/masks/tiles land in <OUTPUT_DIR>/generated/_intermediate,
not in the runtime/ root (owner 10-03: 1841 _INTERMEDIATE_ files piled up there)."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
import config


def test_scratch_path_is_under_generated(tmp_path):
    p = config.scratch_path(tmp_path, "_INTERMEDIATE_x.png")
    assert p.parent == tmp_path / "generated" / "_intermediate" and p.parent.is_dir()


def test_no_writer_drops_scratch_into_the_root():
    bad = []
    for d in ("imaging", "media"):
        for f in os.listdir(os.path.join(ROOT, d)):
            if f.endswith(".py"):
                src = open(os.path.join(ROOT, d, f), encoding="utf-8").read()
                bad += [f"{f}: {m}" for m in re.findall(r'OUTPUT_DIR / f?"_\w+', src)]
    assert not bad, bad
