"""Code files and zips sent with a question are read (live: zip rejected, .py ignored)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot
here = os.path.dirname(os.path.abspath(__file__))
z = tg_bot._extract_doc(os.path.join(here, "..", "bench", "assets", "shop_project.zip"), "p.zip")
assert "shop/money.py" in z and "percent // 10" in z, z[:300]
assert "def test_total" in tg_bot._extract_doc(os.path.join(here, "..", "bench", "assets", "shop_project.zip"), "p.zip")
assert tg_bot._extract_doc(__file__, "x.py").startswith('"""Code files')
print("ok")
