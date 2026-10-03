"""gui_common.open_in_os opens folders on every OS.

Bug: the Memory and Code tabs called os.startfile, which exists only on
Windows -- elsewhere "Open folder" failed with "module 'os' has no attribute
'startfile'".
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

import gui_common  # noqa: E402


@pytest.mark.skipif(sys.platform.startswith("win"), reason="the non-Windows branch")
def test_uses_the_desktop_handler(tmp_path, monkeypatch):
    from PyQt5.QtGui import QDesktopServices
    seen = []
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: seen.append(url) or True))
    gui_common.open_in_os(tmp_path)
    assert seen and seen[0].toLocalFile() == str(tmp_path)
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: False))
    with pytest.raises(OSError):
        gui_common.open_in_os(tmp_path)


def test_no_bare_startfile_left_in_the_gui():
    import glob
    bad = []
    for f in glob.glob(os.path.join(ROOT, "gui", "*.py")):
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            if "os.startfile(" in line and "gui_common.py" not in f:
                src = open(f, encoding="utf-8").read()
                # allowed only behind a Windows check in the same function
                if 'sys.platform.startswith("win")' not in src:
                    bad.append(f"{os.path.basename(f)}:{i}")
    assert not bad, bad
