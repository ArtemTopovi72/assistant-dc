"""A research run must return a MARKDOWN FILE, not a chat full of bubbles.

The result of a 15-60 minute research run is a document. It used to be sent as a
.md attachment only past 6000 characters; anything shorter was pasted into the
chat — and pasted through _md_to_html, so the Markdown was converted AWAY and the
user got a wall of formatted bubbles instead of the document they asked for.

Chat text remains the fallback for a FAILED upload: then it is the only copy that
exists and losing it would be worse than the noise.

Run: venv/Scripts/python.exe tests/test_report_is_markdown_file.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_report_")
import tg_bot as T
T.redirect_data_dir(_DATA)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


CID = 999951
REPORT = ("# Отчёт\n\n## Введение\n\nКороткий отчёт с настоящей разметкой.\n\n"
          "- первый пункт\n- второй пункт\n\n## Вывод\n\nГотово.\n")


def make_bot(upload_ok=True):
    import types
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent, bot.docs = [], []
    bot._send_text = lambda cid, text, **kw: (bot.sent.append(text), 1)[1]
    bot._send_get_id = lambda cid, text, **kw: bot._send_text(cid, text, **kw)

    def _doc(cid, path, caption=""):
        bot.docs.append((path, open(path, encoding="utf-8").read()))
        return upload_ok

    bot._send_document = _doc
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


# ── 1. the threshold no longer hides short reports in the chat ───────────────
print("\n" + "=" * 70)
print("1. EVERY REPORT GOES OUT AS A FILE")
print("=" * 70)
check("the file threshold defaults to 0 (always send a file)",
      T._cfg_int("TG_REPORT_FILE_CHARS", 0) == 0,
      T._cfg_int("TG_REPORT_FILE_CHARS", 0))
check("a short report is longer than the threshold",
      len(REPORT) > T._cfg_int("TG_REPORT_FILE_CHARS", 0))

# ── 2. the attachment is real Markdown, unconverted ──────────────────────────
print("\n" + "=" * 70)
print("2. THE ATTACHMENT IS THE MARKDOWN, VERBATIM")
print("=" * 70)
bot = make_bot(upload_ok=True)
ok = bot._send_report_file(CID, "Тема отчёта", REPORT, "hdr", "ru")
check("the upload is reported as delivered", ok is True, ok)
check("exactly one document was sent", len(bot.docs) == 1, len(bot.docs))
path, body = bot.docs[0]
check("the file is a .md", path.lower().endswith(".md"), path)
check("the Markdown headings survive verbatim",
      "## Введение" in body and "## Вывод" in body, body[:120])
check("the bullets survive as Markdown, not HTML",
      "- первый пункт" in body, body[:200])
check("no HTML was injected into the file",
      "<b>" not in body and "<p>" not in body and "&nbsp;" not in body, body[:200])
check("the topic is carried as the H1", body.lstrip().startswith("# Тема отчёта"),
      body[:60])

# ── 3. a FAILED upload still gives the user the text ─────────────────────────
print("\n" + "=" * 70)
print("3. A FAILED UPLOAD MUST NOT LOSE THE REPORT")
print("=" * 70)
bot2 = make_bot(upload_ok=False)
ok2 = bot2._send_report_file(CID, "Тема", REPORT, "hdr", "ru")
check("a failed upload reports False so the caller falls back", ok2 is False, ok2)

# ── 4. the temp file is cleaned up either way ────────────────────────────────
print("\n" + "=" * 70)
print("4. NO TEMP FILES ARE LEFT BEHIND")
print("=" * 70)
check("the temp directory holding the sent file is gone",
      not os.path.exists(bot.docs[0][0]), bot.docs[0][0])

# ── 5. a weird topic cannot produce an unusable filename ─────────────────────
print("\n" + "=" * 70)
print("5. THE FILENAME IS ALWAYS USABLE")
print("=" * 70)
for topic in ("", "///", "a" * 300, "C:\\evil\\..\\path", "тема: с? *знаками*"):
    b = make_bot(upload_ok=True)
    b._send_report_file(CID, topic, REPORT, "h", "ru")
    if not b.docs:
        check(f"a file was still produced for {topic[:20]!r}", False, "no doc")
        continue
    name = os.path.basename(b.docs[0][0])
    check(f"{topic[:18]!r} -> a safe filename",
          name.endswith(".md") and len(name) < 120
          and not any(c in name for c in '\\/:*?"<>|'),
          name)

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
