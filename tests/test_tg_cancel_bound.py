"""The cancelled-task ids are bounded by forgetting the OLDEST, never the newest.

The bot cleared the whole set once it passed 512 ids, so the 513th cancel
wiped the ids of tasks cancelled moments earlier: a queued task the backend
could not drop (Kafka) then ran anyway.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import tg_bot


def test_recent_ids_evicts_oldest_only():
    ids = tg_bot._RecentIds(3)
    for i in range(5):
        ids.add(i)
    assert len(ids) == 3 and 0 not in ids and 1 not in ids
    assert all(i in ids for i in (2, 3, 4))
    ids.add(2)            # re-adding refreshes it: 3 is now the oldest
    ids.add(5)
    assert 2 in ids and 3 not in ids
    ids.discard(5)
    assert 5 not in ids and len(ids) == 2


def test_recent_cancels_survive_a_long_session():
    ids = tg_bot._RecentIds(512)
    for i in range(2000):
        ids.add(f"t{i}")
    assert len(ids) == 512
    assert all(f"t{i}" in ids for i in range(2000 - 512, 2000))
    assert "t0" not in ids
