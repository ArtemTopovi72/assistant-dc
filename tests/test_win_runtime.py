"""The newest msvcp140.dll wins over PyQt5's bundled 14.26 (live 10-03: a friend's
clone died in «MSVCP140.dll 14.26 ... 0xc0000005» a minute after start)."""
import os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import win_runtime as W


def test_newest_is_picked(monkeypatch):
    v = {"qt/msvcp140.dll": (14, 26, 28720, 3), "sys/msvcp140.dll": (14, 51, 36247, 0),
         "sk/msvcp140.dll": (14, 44, 35211, 0)}
    monkeypatch.setattr(W, "_version", lambda p: v.get(p, ()))
    assert W.newest(list(v)) == "sys/msvcp140.dll"
    assert W.newest(["qt/msvcp140.dll", "sk/msvcp140.dll"]) == "sk/msvcp140.dll"
    assert W.newest([]) == ""


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_launcher_preloads_before_qt():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "launch_all.py"), encoding="utf-8").read()
    assert src.index("preload_newest_msvcp()") < src.index("import assistant") if "import assistant" in src \
        else "preload_newest_msvcp()" in src
