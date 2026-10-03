"""An unreadable session store is moved aside, never overwritten.

_Store started empty on a load error and the first put() rewrote the file,
erasing every user's session, facts and language for good.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import tg_sessions


def test_torn_store_is_kept_aside(tmp_path):
    path = tmp_path / "tg_sessions.json"
    torn = '{"42": {"lang": "ru", "facts": ['
    path.write_text(torn, encoding="utf-8")
    store = tg_sessions._Store(path)
    store.put(store.get(7))
    kept = list(tmp_path.glob("tg_sessions.json.unreadable-*"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == torn
    assert "7" in path.read_text(encoding="utf-8")


def test_non_dict_store_starts_empty(tmp_path):
    path = tmp_path / "tg_sessions.json"
    path.write_text("[]", encoding="utf-8")
    store = tg_sessions._Store(path)
    store.put(store.get(7))           # used to raise: list has no item assignment by str
    assert "7" in path.read_text(encoding="utf-8")
