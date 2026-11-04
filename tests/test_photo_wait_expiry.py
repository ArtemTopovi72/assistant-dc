"""A 🎭 Style pressed and forgotten made the bot take a news screenshot, sent much later,
for the style reference and restyle the earlier picture (live 2026-10-18). The wait for
«the next photo» expires.

Run: venv/Scripts/python.exe tests/test_photo_wait_expiry.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T

S = T._Session


def test_a_forgotten_style_wait_is_disarmed():
    s = S(1)
    s.pending_style_target = "img1"
    s.awaiting_animate_photo = True
    s.__dict__["_wait_ts"]["pending_style_target"] = time.time() - S.PHOTO_WAIT_TTL_S - 5
    assert s.expire_photo_waits() is True
    assert s.pending_style_target == ""
    assert s.awaiting_animate_photo is True        # armed a moment ago, stays


def test_a_fresh_wait_stays_armed():
    s = S(1)
    s.pending_outfit_target = "img2"
    assert s.expire_photo_waits() is False
    assert s.pending_outfit_target == "img2"


def test_restored_wait_keeps_original_age():
    s = S(1)
    s.pending_style_target = 'img3'
    s.__dict__['_wait_ts']['pending_style_target'] = time.time() - S.PHOTO_WAIT_TTL_S - 5
    restored = S(1, s.to_dict())
    assert restored.expire_photo_waits() is True
    assert restored.pending_style_target == ''


def test_recent_restored_wait_is_preserved():
    s = S(1)
    s.pending_swap_target = 'img4'
    restored = S(1, s.to_dict())
    assert restored.expire_photo_waits() is False
    assert restored.pending_swap_target == 'img4'


def test_legacy_wait_without_timestamp_is_not_revived():
    s = S(1, {'pending_swap_target': 'old', 'swap_where': 'first'})
    assert s.expire_photo_waits() is True
    assert not s.pending_swap_target
