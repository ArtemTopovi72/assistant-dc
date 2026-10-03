"""A message the user forwards from their OWN Saved Messages is their own
instruction, not "someone else's words" (live 2026-09-14: a collage request
+ DCIM.zip self-forwarded was acknowledged instead of done)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T


def test_self_forward_is_not_a_forward():
    m = {"from": {"id": 5}, "forward_origin": {"type": "user", "sender_user": {"id": 5}}, "text": "x"}
    assert not T._is_forwarded(m)
    m = {"from": {"id": 5}, "forward_from": {"id": 5}, "forward_date": 1, "text": "x"}
    assert not T._is_forwarded(m)


def test_someone_elses_forward_still_is():
    m = {"from": {"id": 5}, "forward_origin": {"type": "user", "sender_user": {"id": 9}}, "text": "x"}
    assert T._is_forwarded(m)
    assert T._is_forwarded({"from": {"id": 5}, "forward_origin": {"type": "hidden_user", "sender_user_name": "A"}})
    assert T._is_forwarded({"from": {"id": 5}, "forward_from_chat": {"id": -100}, "forward_date": 1})
    assert not T._is_forwarded({"from": {"id": 5}, "text": "plain"})


if __name__ == "__main__":
    test_self_forward_is_not_a_forward()
    test_someone_elses_forward_still_is()
    print("2/2 passed")
