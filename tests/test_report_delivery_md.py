"""A long report goes out ONCE, as a file — and its headings keep their newlines.

Two live complaints (2026-07-31, screenshot):

A. A 38k-character deep-research report was delivered as "Почему говно не
   тонет.md" AND then pasted into the chat as a dozen bubbles. The attachment is
   the report; repeating it buries the conversation.

B. The chat copy rendered the title and the next heading fused into one word:
   "…Gastrointestinal PhysiologyAbstract". Cause: the ATX-heading rule ended in
   `\\s*#*\\s*$`, and `\\s` matches NEWLINES, so it consumed the line break after
   every heading plus the blank line following it.

Run: venv/Scripts/python.exe tests/test_report_delivery_md.py
"""
import os, sys, re, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T

# TelegramBot is split across a dozen mixin modules, and the split keeps
# moving -- this list was already several modules out of date. Source-level
# checks must read ALL of them or moving a method makes the assertion vacuous
# instead of red, so read the family off disk rather than naming its members.
import glob as _glob
def _bot_source():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return chr(10).join(open(_p, encoding="utf-8").read()
                        for _p in sorted(_glob.glob(os.path.join(_root, "bot", "tg_*.py"))))

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}\n         {detail}")


# ---------------------------------------------------- B. the markdown itself
print("\nHEADINGS KEEP THE LINE BREAK THAT FOLLOWS THEM")
md = ("# Mechanisms of Fecal Buoyancy: Analysis of Gastrointestinal Physiology\n"
      "\n"
      "## Abstract\n"
      "\n"
      "Fecal buoyancy is a complex phenomenon.\n")
out = T._md_to_html(md)
check("the two headings are NOT fused into one word",
      "</b><b>" not in out, out[:160])
check("'Physiology' and 'Abstract' are separated",
      "PhysiologyAbstract" not in T._html_to_plain(out),
      T._html_to_plain(out)[:160])
check("the blank line between them survives", "</b>\n\n<b>" in out, repr(out[:170]))
check("a heading is still bold", "<b>Abstract</b>" in out, out[:160])
check("the body text is still there", "complex phenomenon" in out)

check("a lone heading still renders",
      T._md_to_html("## Only heading") == "<b>Only heading</b>",
      repr(T._md_to_html("## Only heading")))
check("closing hashes are still stripped",
      T._md_to_html("### Sub ###").strip() == "<b>Sub</b>",
      repr(T._md_to_html("### Sub ###")))
check("an indented heading still works",
      "<b>Indented</b>" in T._md_to_html("   ## Indented"),
      repr(T._md_to_html("   ## Indented")))
check("a '#' that is not a heading is left alone",
      "<b>" not in T._md_to_html("issue #42 was fixed"),
      T._md_to_html("issue #42 was fixed"))
check("consecutive headings each keep their own line",
      T._md_to_html("# A\n## B\n### C").count("\n") == 2,
      repr(T._md_to_html("# A\n## B\n### C")))

# Bullets and links must not have regressed with the heading change.
check("bullets still render", "• one" in T._md_to_html("- one"),
      T._md_to_html("- one"))
check("links still render",
      '<a href="https://x.io">t</a>' in T._md_to_html("[t](https://x.io)"),
      T._md_to_html("[t](https://x.io)"))

# ------------------------------------------- A. delivered once, not twice
print("\nA LONG REPORT IS NOT PASTED INTO THE CHAT AS WELL")
src = _bot_source()
m = re.search(r"sent_as_file = False(.*?)# ── end deep research", src, re.S)
body = m.group(1) if m else ""
check("the deep-research send block was found", bool(body.strip()),
      "anchor moved — the checks below would be vacuous")
if not body.strip():
    print("\nABORT: cannot verify the send path")
    sys.exit(1)

check("the file result is captured, not discarded",
      "sent_as_file = self._send_report_file" in body, body[:200])
check("when the file lands, only the header is sent",
      re.search(r"if sent_as_file:.*?self\._send_text\(chat_id, header", body, re.S)
      is not None, body[:400])
check("the full report is NOT chunked in that branch",
      "_split_html" not in body.split("else:")[0], body.split("else:")[0][-300:])
check("but a failed upload still sends the text — it is the only copy",
      "_split_html" in body and "else:" in body, body[-400:])

sig = inspect.getsource(T.TelegramBot._send_report_file) if hasattr(T, "TelegramBot") else ""
if not sig:
    cls = next((v for v in vars(T).values()
                if isinstance(v, type) and hasattr(v, "_send_report_file")), None)
    sig = inspect.getsource(cls._send_report_file) if cls else ""
check("_send_report_file reports whether it actually delivered",
      "-> bool" in sig and "return True" in sig and "return False" in sig, sig[:200])

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
