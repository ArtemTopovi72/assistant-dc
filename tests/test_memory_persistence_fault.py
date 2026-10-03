"""Regression: the memory subsystem must survive corrupt/interrupted persistence
without crashing the assistant.

BUG #11 (found by filesystem/JSON fault injection): a corrupted session_memory.json
(a JSON object/scalar, or a list containing non-dict entries) loaded non-dict items
into session_memory, and Context.memory_text() — called on EVERY turn — then crashed
with `AttributeError: 'str' object has no attribute 'get'`, bricking that memory
profile until the file was manually deleted. facts.json already filtered to dicts;
session load did not. Fix: filter session items to dicts on load + a defensive
isinstance guard in memory_text.

Run: venv/Scripts/python.exe tests/test_memory_persistence_fault.py
"""
import os, sys, threading, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from models import Context, _atomic_write_json


def _ctx(d):
    return Context(models=None, transcription_cache={}, cache_file=d / "c.json",
                   asr_lock=threading.Lock(), tts_lock=threading.Lock(), model_name="m")


def test_corrupt_session_memory_shapes():
    d = Path(tempfile.mkdtemp())
    # The last shape mixes valid dicts with junk. It uses kind="fact" because
    # load_memory now restores only DURABLE kinds across sessions — with a
    # non-durable kind the dict filter would never be exercised and the check
    # would pass for the wrong reason.
    shapes = ['{"x":1}', '["a","b",123]', '42', 'null', 'not json at all',
              '[{"text":"good","kind":"fact"}, "junk", 7, {"text":"also","kind":"fact"}]']
    for shp in shapes:
        (d / "session_memory.json").write_text(shp, encoding="utf-8")
        c = _ctx(d)
        c.load_memory(d)                 # must not raise
        _ = c.memory_text()              # must not raise (the every-turn call)
        for it in c.session_memory:
            assert isinstance(it, dict), f"non-dict survived load from {shp!r}: {it!r}"
    # the last shape keeps exactly the two valid dict items
    assert len(c.session_memory) == 2, f"expected 2 valid items, got {len(c.session_memory)}"
    print("PASS corrupt session_memory.json shapes never crash memory_text (BUG #11)")

    # Pin the durability contract itself: in-session chatter ("assistant",
    # "generate", "search", untyped items) must NOT come back on the next start,
    # or an old unrelated turn resurfaces in a fresh conversation.
    (d / "session_memory.json").write_text(json.dumps([
        {"text": "keep me", "kind": "fact"},
        {"text": "old reply", "kind": "assistant"},
        {"text": "old image prompt", "kind": "generate"},
        {"text": "no kind at all"},
    ]), encoding="utf-8")
    c = _ctx(d)
    c.load_memory(d)
    kept = [i.get("text") for i in c.session_memory]
    assert kept == ["keep me"], f"non-durable memory survived a restart: {kept}"
    print("PASS only durable memory kinds survive a restart")


def test_corrupt_facts_and_summary():
    d = Path(tempfile.mkdtemp())
    (d / "facts.json").write_text('{"nope": true}', encoding="utf-8")
    (d / "summary.json").write_text('garbage', encoding="utf-8")
    c = _ctx(d)
    c.load_memory(d)                     # must not raise
    _ = c.facts_text()
    assert all(isinstance(f, dict) for f in c.pinned_facts)
    print("PASS corrupt facts.json / summary.json load without crashing")


def test_atomic_write_survives_replace_permissionerror(monkeypatch=None):
    import models as models_mod
    d = Path(tempfile.mkdtemp())
    calls = {"n": 0}
    real_replace = os.replace
    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:               # fail twice, succeed on the 3rd (retry path)
            raise PermissionError("WinError 32: file in use")
        return real_replace(src, dst)
    os.replace = flaky_replace
    try:
        _atomic_write_json(d / "x.json", {"a": 1})
        assert json.loads((d / "x.json").read_text(encoding="utf-8")) == {"a": 1}
    finally:
        os.replace = real_replace
    assert calls["n"] == 3, f"retry path not exercised ({calls['n']})"
    # no temp files left behind
    assert not list(d.glob("*.tmp")), "atomic write left a temp file behind"
    print("PASS _atomic_write_json retries on PermissionError and cleans up temp")


def test_save_memory_never_raises_on_bad_dir():
    d = Path(tempfile.mkdtemp())
    c = _ctx(d)
    c.remember("note", "hello")
    # point at a path whose parent is a FILE -> mkdir fails; save_memory must swallow it
    bad_parent = d / "afile"
    bad_parent.write_text("x", encoding="utf-8")
    c.save_memory(bad_parent / "sub")    # must not raise
    print("PASS save_memory swallows filesystem errors (no crash)")


def test_roundtrip_preserves_facts_and_session():
    d = Path(tempfile.mkdtemp())
    c = _ctx(d)
    c.remember("fact", "durable session item")
    c.remember("note", "in-session chatter")
    c.pin_fact("durable fact A")
    c.save_memory(d)
    c2 = _ctx(d)
    c2.load_memory(d)
    # save_memory writes only kind="fact"; a "note" is in-session noise by design.
    assert any("durable session item" in str(i.get("text")) for i in c2.session_memory)
    assert not any("in-session chatter" in str(i.get("text")) for i in c2.session_memory)
    assert any("durable fact A" in str(f.get("text")) for f in c2.pinned_facts)
    print("PASS save/load round-trip preserves durable memory and facts")


if __name__ == "__main__":
    test_corrupt_session_memory_shapes()
    test_corrupt_facts_and_summary()
    test_atomic_write_survives_replace_permissionerror()
    test_save_memory_never_raises_on_bad_dir()
    test_roundtrip_preserves_facts_and_session()
    print("\ndone")
