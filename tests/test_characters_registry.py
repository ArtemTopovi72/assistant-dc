"""The character registry, pinned at the points where the two surfaces meet.

Every check here is a defect that would otherwise show up as the bot offering a
persona it cannot draw, or as the GUI and the bot disagreeing about a checkbox.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import characters as C


def _fresh():
    d = tempfile.mkdtemp(prefix="chars_")
    C.redirect_store(Path(d) / "characters.json")
    return Path(d)


def test_slug_transliterates_russian():
    # Two different Russian names must not land on the same id.
    assert C.slugify("НейроСтепан") == "neyrostepan"
    assert C.slugify("Тёмный Эльф") != C.slugify("Светлый Эльф")


def test_enabled_but_missing_adapter_is_not_offered():
    """The whole point of settling against disk."""
    _fresh()
    C.upsert("hero", name="Hero", status="ready",
             lora_file="Z:/nowhere/hero.safetensors", tg_enabled=True)
    rec = C.get("hero")
    assert rec["status"] == "failed", "a vanished adapter must not stay 'ready'"
    assert rec["tg_available"] is False
    assert C.telegram_characters() == []


def test_enabled_with_real_adapter_is_offered():
    d = _fresh()
    lora = d / "hero.safetensors"
    lora.write_bytes(b"x")
    C.upsert("hero", status="ready", lora_file=str(lora), tg_enabled=True)
    assert [c["slug"] for c in C.telegram_characters()] == ["hero"]
    C.set_tg_enabled("hero", False)
    assert C.telegram_characters() == []


def test_upsert_keeps_untouched_fields():
    _fresh()
    C.upsert("hero", name="Hero", trigger="hero_tok")
    C.upsert("hero", status="training")
    rec = C.get("hero")
    assert rec["trigger"] == "hero_tok" and rec["status"] == "training"


def test_bad_input_is_refused():
    _fresh()
    for bad in ("", "Hero", "with space", "..", "x" * 65):
        try:
            C.upsert(bad)
        except ValueError:
            continue
        raise AssertionError("accepted bad id: %r" % (bad,))
    try:
        C.upsert("hero", status="almost")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an invented status")


def test_corrupt_store_does_not_take_the_list_down():
    d = _fresh()
    (d / "characters.json").write_text("{ not json", encoding="utf-8")
    assert C.list_characters() == []
    C.upsert("hero")                      # and it recovers by rewriting
    assert [c["slug"] for c in C.list_characters()] == ["hero"]


def test_delete_leaves_the_dataset_alone():
    d = _fresh()
    ds = d / "ds"
    ds.mkdir()
    (ds / "a.jpg").write_bytes(b"x")
    C.upsert("hero", dataset_dir=str(ds))
    assert C.delete("hero") is True
    assert C.delete("hero") is False
    assert (ds / "a.jpg").exists(), "deleting a record must not delete photos"


def test_liveness_survives_a_silent_download_phase(tmp_path, monkeypatch):
    """Measured, not assumed: a cold run downloads a 12 GB base model and
    writes nothing to the log for many minutes. A freshness-only check called
    that healthy run dead, so liveness follows the PROCESS."""
    import lora_training as LT
    monkeypatch.setattr(LT, "TRAIN_ROOT", tmp_path)
    (tmp_path / "hero").mkdir()
    old_log = tmp_path / "hero" / "train.log"
    old_log.write_text("Loading transformer", encoding="utf-8")
    import os, time as _t
    stale = _t.time() - 3600
    os.utime(old_log, (stale, stale))          # an hour of silence

    class _P:
        def __init__(self, line):
            self.info = {"cmdline": line.split()}
    fake = type("psutil", (), {"process_iter": staticmethod(
        lambda attrs=None: [_P("python run.py runtime/lora_training/hero/hero.yaml")])})
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
    assert LT.is_training("hero") is True, "a downloading run must not read as dead"
