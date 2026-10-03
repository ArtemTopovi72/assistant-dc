import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_reply_shape as R

src = [{"domain": "xn----8sbhycugqd1i.xn--p1ai", "url": "https://xn----8sbhycugqd1i.xn--p1ai/a", "title": "Гора Эльбрус"},
       {"domain": "climbingbrothers.org", "url": "https://climbingbrothers.org/e", "title": "Высота"}]
out = R.shape_search_reply("Высота 5642 м (xn----8sbhycugqd1i.xn--p1ai; climbingbrothers.org).", ["высота Эльбруса"], src)
assert "Высота 5642 м [1] [2]." in out, out
assert "xn--" not in out.split("🔗")[1].replace('href="https://xn----8sbhycugqd1i.xn--p1ai/a"', ""), out
print("PASS punycode domains cited and shown in Cyrillic")
