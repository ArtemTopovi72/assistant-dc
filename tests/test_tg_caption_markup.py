"""Media captions were the one Telegram surface with no markup safety net.

sendMessage has had two guards for a long time: _split_html cuts on tag-safe
boundaries, and _post_message resends as plain text when Telegram answers
"can't parse entities". The media senders had NEITHER — they did a blind
`caption[:1024]` with parse_mode=HTML and retried the identical body three
times before giving up.

That loses the payload, not the formatting: a research report goes out as a
.md document with a caption, and a rendered clip costs minutes of GPU. A cut
landing inside <a href="..."> or between <b> and </b> made Telegram reject the
whole request, so the user got nothing at all.

Run: venv/Scripts/python.exe tests/test_tg_caption_markup.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="caption_"))

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


LIMIT = T._MAX_CAPTION


def _tags_balanced(s: str) -> bool:
    """Every opened formatting tag is closed, in order — Telegram's actual rule."""
    stack = []
    for m in T._TAG_RE.finditer(s):
        closing, name = m.group(1), m.group(2).lower()
        if closing:
            if not stack or stack[-1] != name:
                return False
            stack.pop()
        else:
            stack.append(name)
    return not stack


def _no_partial_tag(s: str) -> bool:
    """No '<' after the last '>' — i.e. the string never ends mid-tag."""
    return s.rfind("<") <= s.rfind(">")


print("=" * 70)
print("1. Short captions are passed through untouched")
print("=" * 70)
for s in ("", "hello", "<b>bold</b> and <i>italic</i>", "a" * LIMIT):
    check(f"unchanged: {s[:24]!r}", T._safe_caption(s) == s, repr(T._safe_caption(s)[:40]))

print()
print("=" * 70)
print("2. Long captions are cut to the limit, tag-safe and balanced")
print("=" * 70)
# Each case puts the 1024th character somewhere hostile.
CASES = {
    "cut lands inside an <a href> attribute":
        "x" * (LIMIT - 10) + '<a href="https://example.com/some/long/path">link</a>' + "y" * 200,
    "cut lands between <b> and </b>":
        "<b>" + "x" * (LIMIT + 200) + "</b>",
    "cut lands inside a tag NAME":
        "x" * (LIMIT - 2) + "<code>" + "y" * 200 + "</code>",
    "nested tags open across the cut":
        "<b><i><u>" + "x" * (LIMIT + 100) + "</u></i></b>",
    "many short tags, cut anywhere":
        ("<b>a</b><i>b</i>" * 200),
    "an attribute containing spaces (a naive space-cut trap)":
        "x" * (LIMIT - 30) + '<a href="https://e.com/a b c d e f">t</a>' + "z" * 300,
}
for name, raw in CASES.items():
    out = T._safe_caption(raw)
    check(f"{name}: within Telegram's {LIMIT}", len(out) <= LIMIT, len(out))
    check(f"{name}: never ends mid-tag", _no_partial_tag(out), repr(out[-60:]))
    check(f"{name}: tags balanced", _tags_balanced(out), repr(out[-80:]))
    check(f"{name}: kept real content", len(out) > LIMIT // 2, len(out))

print()
print("=" * 70)
print("3. A tag larger than the whole budget degrades instead of emitting junk")
print("=" * 70)
monster = '<a href="' + "u" * (LIMIT + 500) + '">t</a>'
out = T._safe_caption(monster)
check("still within the limit", len(out) <= LIMIT, len(out))
check("no partial tag survives", _no_partial_tag(out), repr(out[:60]))
check("markup was dropped rather than half-written", "<a" not in out, repr(out[:60]))

print()
print("=" * 70)
print("4. _is_caption_parse_error fires ONLY on a formatting rejection")
print("=" * 70)
class _R:
    def __init__(self, code, desc=None, boom=False):
        self.status_code = code; self._d = desc; self._boom = boom
        # The senders log r.text on a non-200; without it the AttributeError is
        # swallowed by the retry loop's `except Exception` and the degrade path
        # is never reached — which is exactly how this fixture first lied.
        self.text = str(desc or "")
    def json(self):
        if self._boom: raise ValueError("not json")
        return {"ok": False, "description": self._d}

check("200 is never a parse error", not T._is_caption_parse_error(_R(200)))
check("can't parse entities -> True",
      T._is_caption_parse_error(_R(400, "Bad Request: can't parse entities")))
check("unclosed tag -> True",
      T._is_caption_parse_error(_R(400, "Bad Request: Unclosed start tag")))
# These must NOT trigger a resend — the request would fail again identically,
# and a blocked bot resent three more times is just noise.
check("bot blocked -> False", not T._is_caption_parse_error(_R(403, "bot was blocked by the user")))
check("file too big -> False", not T._is_caption_parse_error(_R(413, "Request Entity Too Large")))
check("chat not found -> False", not T._is_caption_parse_error(_R(400, "chat not found")))
check("a non-JSON body does not raise", not T._is_caption_parse_error(_R(500, boom=True)))

print()
print("=" * 70)
print("5. A rejected caption never costs the user the FILE")
print("=" * 70)
# Drive the real _send_document against a fake Telegram that rejects any HTML
# caption. Nothing here touches the network: only requests.post is replaced.
import types

class _FakeTG:
    def __init__(self, limit_reject=True):
        self.calls = []
        self.limit_reject = limit_reject
    def post(self, url, data=None, files=None, timeout=None):
        self.calls.append(dict(data or {}))
        if self.limit_reject and (data or {}).get("parse_mode") == "HTML":
            return _R(400, "Bad Request: can't parse entities in message caption")
        return _R(200)

bot = T.TelegramBot.__new__(T.TelegramBot)      # no network, no polling thread
bot._api = "http://127.0.0.1:0/botTEST"

doc = os.path.join(tempfile.mkdtemp(prefix="capdoc_"), "report.md")
open(doc, "w", encoding="utf-8").write("# report\n")

_real_post = T.requests.post
fake = _FakeTG()
T.requests.post = fake.post
_real_sleep = T.time.sleep
T.time.sleep = lambda *_a, **_k: None
try:
    sent = bot._send_document(123, doc, caption="<b>Отчёт</b> " + "x" * 2000)
finally:
    T.requests.post = _real_post
    T.time.sleep = _real_sleep

check("the document was delivered despite the caption rejection", sent, sent)
check("it retried with the markup stripped",
      any("parse_mode" not in c for c in fake.calls), [list(c) for c in fake.calls])
check("the plain retry still carried the text",
      any("parse_mode" not in c and "Отчёт" in (c.get("caption") or "")
          for c in fake.calls),
      [(c.get("parse_mode"), (c.get("caption") or "")[:20]) for c in fake.calls])
check("the plain caption respects the limit",
      all(len(c.get("caption") or "") <= LIMIT for c in fake.calls),
      [len(c.get("caption") or "") for c in fake.calls])
# The degrade must not eat a transport retry: it is our own bad body, not a
# flaky network, so the file still gets its full retry budget afterwards.
check("degrading did not consume the retry budget",
      len(fake.calls) >= 2, len(fake.calls))

print()
print("=" * 70)
print("5b. The senders actually USE _safe_caption (not caption[:1024])")
print("=" * 70)
# Sections 1-3 prove the helper is correct; they say nothing about whether the
# senders call it. Reverting the two call sites to a blind slice left this suite
# fully green until these checks existed — assert on the body that goes ON THE
# WIRE, from a caption whose 1024th character sits inside an <a href>.
hostile = "x" * (LIMIT - 10) + '<a href="https://example.com/a/very/long/path">t</a>' + "y" * 300
for label, sender, kwargs in (
        ("sendDocument", lambda **kw: bot._send_document(123, doc, **kw), {}),
):
    accept = _FakeTG(limit_reject=False)          # accept everything: inspect the body
    T.requests.post = accept.post
    T.time.sleep = lambda *_a, **_k: None
    try:
        sender(caption=hostile, **kwargs)
    finally:
        T.requests.post = _real_post
        T.time.sleep = _real_sleep
    wire = (accept.calls[0].get("caption") or "") if accept.calls else ""
    check(f"{label}: caption within the limit", len(wire) <= LIMIT, len(wire))
    check(f"{label}: caption on the wire is not cut mid-tag",
          _no_partial_tag(wire), repr(wire[-70:]))
    check(f"{label}: caption on the wire has balanced tags",
          _tags_balanced(wire), repr(wire[-70:]))

print()
print("=" * 70)
print("6. A NON-formatting failure is not retried forever with a plain caption")
print("=" * 70)
class _Blocked(_FakeTG):
    def post(self, url, data=None, files=None, timeout=None):
        self.calls.append(dict(data or {}))
        return _R(403, "Forbidden: bot was blocked by the user")

blocked = _Blocked()
T.requests.post = blocked.post
T.time.sleep = lambda *_a, **_k: None
try:
    sent2 = bot._send_document(123, doc, caption="<b>hi</b>")
finally:
    T.requests.post = _real_post
    T.time.sleep = _real_sleep
check("a blocked bot is reported as failure", not sent2, sent2)
check("...and it stopped at the normal retry budget",
      len(blocked.calls) == T._API_RETRIES, len(blocked.calls))
check("...without ever stripping the markup (that was not the problem)",
      all(c.get("parse_mode") == "HTML" for c in blocked.calls),
      [c.get("parse_mode") for c in blocked.calls])

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
