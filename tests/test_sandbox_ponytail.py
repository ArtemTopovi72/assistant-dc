"""Ponytail rules ride along coding turns in the sandbox, and only those."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ask_stub  # noqa: F401  the model's read, stubbed
import graph_compose as G


class Box: sandbox = object()
class NoBox: sandbox = None


def test_coding_turn_gets_rules():
    b = G._ponytail_block(Box(), "напиши скрипт, который считает строки в csv")
    assert "stdlib" in b and "root cause" in b and "(full)" in b


def test_non_coding_and_no_sandbox_get_nothing():
    for t in ("нарисуй кота на крыше", "какая погода в Казани", "убери надпись"):
        assert G._ponytail_block(Box(), t) == "", t
    assert G._ponytail_block(NoBox(), "fix the bug in main.py") == ""


def test_levels_and_off():
    os.environ["PONYTAIL"] = "ultra"
    try:
        assert "YAGNI extremist" in G._ponytail_block(Box(), "refactor this module")
        os.environ["PONYTAIL"] = "off"
        assert G._ponytail_block(Box(), "refactor this module") == ""
    finally:
        os.environ.pop("PONYTAIL", None)


if __name__ == "__main__":
    test_coding_turn_gets_rules(); test_non_coding_and_no_sandbox_get_nothing(); test_levels_and_off(); print("3/3 ok")
