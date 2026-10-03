"""Sandbox file names in a reply become t.me deep links; the token finds the file."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_sandbox_links as L


class Box:
    def __init__(self, root):
        self.root = root


def test_names_become_links_and_resolve(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "app.py").write_text("x")
    (tmp_path / "out" / "report.csv").write_text("x")
    box = Box(str(tmp_path))
    html = "Готово: <code>app.py</code> и out/report.csv, а readme.md нет.<pre>app.py</pre>"
    got = L.linkify(html, 42, box, "mybot")
    assert f'<a href="https://t.me/mybot?start={L.token(42, "app.py")}">app.py</a>' in got
    assert f'?start={L.token(42, "out/report.csv")}">out/report.csv</a>' in got
    assert "<pre>app.py</pre>" in got and "readme.md нет" in got
    assert "<code><a" not in got
    assert L.find(box, 42, L.token(42, "out/report.csv")) == "out/report.csv"
    assert L.find(box, 7, L.token(42, "app.py")) is None      # another user's link finds nothing


def test_no_sandbox_files_no_change(tmp_path):
    assert L.linkify("app.py", 1, Box(str(tmp_path)), "bot") == "app.py"
