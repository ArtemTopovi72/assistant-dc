"""Regression tests for the 📚 Documents fixes (upload/index path in tg_bot.py
and Library.build dedup in library.py).

Offline: redirect_data_dir to a fresh tempdir, stub the network (_api_post
recorder) and the embedding layer (reports the endpoint DOWN — no LM Studio,
no GPU). The real TelegramBot._index_document and the real library.Library /
knowledge.KnowledgeBase run against a throwaway SQLite file.

Covers, worst first (see scratchpad/audit_09_library.md):
  1. pending_prefix leaking across an upload
  2. unreadable/empty upload WITH a caption silently answered as if read
  3. illegal-character filenames + raw temp-path leak in the error
  4. duplicate documents on re-upload
  5. My documents list: 40-item cap with no note + blind 4096 slice
  6. per-file passage count vs whole-library total
"""
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import pathlib
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tg_bot as T

DATA = tempfile.mkdtemp(prefix="libfix_test_")
T.redirect_data_dir(DATA)

import knowledge  # noqa: E402  (after redirect_data_dir)


class _StubEmbedder:
    """No network, no GPU: reports the embed endpoint as DOWN (documented
    graceful-degradation path). FTS-only indexing still runs for real."""
    def __init__(self, *a, **k):
        self.model = "stub"; self.base_url = "stub"; self.enabled = False
        self.dim = None
    def available(self, recheck=30.0):
        return False
    def embed(self, texts, **k):
        return [None] * len(texts)
    def embed_one(self, text, **k):
        return None


knowledge.Embedder = _StubEmbedder

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1
        print(f"PASS  {name}")
    else:
        BAD += 1
        print(f"FAIL  {name}   {extra}")


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {},
                        silent_mode=True)
    bot.posts = []

    def _api_post(method, p=None, **k):
        bot.posts.append((method, p or {}))
        return {"ok": True, "result": {"message_id": len(bot.posts)}}

    bot._api_post = _api_post
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    return bot


def sent(bot):
    return [p["text"] for m, p in bot.posts if m == "sendMessage"]


def last(bot):
    s = sent(bot)
    return s[-1] if s else ""


def fresh(bot, cid, lang="en"):
    s = bot._get_session(cid)
    s.clear_context(); s.reg_state = ""; s.lang = lang
    s.use_docs = False; s.menu = ""; s.pending_prefix = ""
    bot._store.put(s)
    bot.posts.clear()
    return s


def upload(bot, cid, fname, data, caption=""):
    bot._dl_bytes = lambda fid: data
    bot._resolve_and_push(cid, [{"type": "document", "file_id": "f",
                                 "filename": fname, "caption": caption}])


TXT = ("Lorem ipsum dolor sit amet. " * 40).encode()

CID = 881001


# ── 1. pending_prefix survives an upload ────────────────────────────────────
print("=" * 70); print("1. pending_prefix cleared on upload")
bot = make_bot(); bot._backend = T.InMemoryBackend()
s = fresh(bot, CID)
s.pending_prefix = "generate an image of: "
bot._store.put(s)
upload(bot, CID, "notes.txt", TXT)
check("upload clears the stale pending_prefix",
      bot._get_session(CID).pending_prefix == "",
      repr(bot._get_session(CID).pending_prefix))
check("an upload does not switch into the library menu (owner 10-03)", bot._get_session(CID).menu != "library")

bot.posts.clear(); bot._backend = T.InMemoryBackend()
bot._resolve_and_push(CID, [{"type": "text", "text": "what does it say about revenue?"}])
check("the next question is NOT glued to the old Draw prefix",
      bot._backend.depth() == 1 and
      not bot._backend.service_order()[0].user_text.startswith("generate an image of:"),
      bot._backend.service_order()[0].user_text if bot._backend.depth() else "nothing queued")


# ── 2. unreadable/empty upload WITH a caption ───────────────────────────────
print("=" * 70); print("2. unreadable upload + caption is reported, not hallucinated")
bot = make_bot(); bot._backend = T.InMemoryBackend(); fresh(bot, CID)
upload(bot, CID, "book.pdf", None, caption="summarise chapter 3")
check("a failed download reports doc_unreadable even with a caption",
      any("could not read" in t.lower() or "❓" in t for t in sent(bot)), sent(bot))
check("the caption is NOT queued to the agent alone",
      bot._backend.depth() == 0, bot._backend.depth())

bot.posts.clear(); bot._backend = T.InMemoryBackend()
upload(bot, CID, "empty.pdf", b"", caption="summarise chapter 3")
check("a zero-byte download ALSO reports doc_unreadable with a caption",
      any("could not read" in t.lower() or "❓" in t for t in sent(bot)), sent(bot))
check("zero-byte + caption is not silently queued either",
      bot._backend.depth() == 0, bot._backend.depth())


# ── 2b. unsupported file type (forwarded video/GIF sent as "document") ─────
# Live bug report: a forwarded animation (yappi.mp4) sent as a Telegram
# document hit the full "Indexing... this may take a minute" flow and then
# failed with a bare, meaningless "Could not index yappi.mp4: RuntimeError" --
# library.ingest() has never supported video, so this was never going to
# work. It should be rejected up front with an explanation, not attempted.
print("=" * 70); print("2b. unsupported file type is rejected up front, not attempted")
bot = make_bot(); bot._backend = T.InMemoryBackend(); fresh(bot, CID)
upload(bot, CID, "yappi.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)
check("an unsupported extension gets a clear, translated explanation",
      any("isn't a document i can add" in t.lower() for t in sent(bot)), sent(bot))
check("...never the bare exception-class-name message",
      not any("runtimeerror" in t.lower() for t in sent(bot)), sent(bot))
check("...and never even shows the 'Indexing...' message first",
      not any("indexing" in t.lower() for t in sent(bot)), sent(bot))
check("nothing was queued to the agent either",
      bot._backend.depth() == 0, bot._backend.depth())
check("the file was NOT added to the library",
      "yappi.mp4" not in {d.get("title") for d in bot._library_stats(CID)["docs"]},
      bot._library_stats(CID))

bot.posts.clear(); bot._backend = T.InMemoryBackend()
upload(bot, CID, "photo.jpg", b"\xff\xd8\xff" + b"\x00" * 200)
# Since 2026-09-28 a picture sent "as a file" goes the photo way (receipt.jpg
# as a document was answered «text could not be extracted», live).
check("an image sent as a document is not rejected as a document",
      not any("isn't a document i can add" in t.lower() for t in sent(bot)), sent(bot))
check("...it is queued as a photo task", bot._backend.depth() == 1, bot._backend.depth())

bot.posts.clear(); bot._backend = T.InMemoryBackend()
upload(bot, CID, "notes.txt", TXT)
check("a genuinely supported extension is unaffected by the new gate",
      any("indexed" in t.lower() for t in sent(bot)), sent(bot))


# ── 3. illegal filename characters + no raw path leak ───────────────────────
print("=" * 70); print("3. illegal filename chars index cleanly, no path leak")
bot = make_bot(); bot._backend = T.InMemoryBackend(); fresh(bot, CID)
for bad_name in ["report<v2>.txt", 'quote".txt', "a|b.txt", "what?.md"]:
    bot.posts.clear()
    upload(bot, CID, bad_name, TXT)
    ok = any("✅" in t or "indexed" in t for t in sent(bot))
    check(f"{bad_name!r} indexes successfully now",
          ok, sent(bot))
    check(f"{bad_name!r}: no message leaks a filesystem path",
          all(("AppData" not in t and "tgdoc_" not in t and "C:\\" not in t)
              for t in sent(bot)),
          sent(bot))
    if ok:
        st = bot._library_stats(CID)
        titles = [d["title"] for d in st["docs"]]
        check(f"{bad_name!r}: display title kept the original name",
              bad_name in titles, titles)


# ── 4. re-upload replaces instead of duplicating ────────────────────────────
print("=" * 70); print("4. re-upload does not duplicate")
CID4 = 881004
bot = make_bot(); bot._backend = T.InMemoryBackend(); fresh(bot, CID4)
upload(bot, CID4, "dup.txt", TXT)
n1 = bot._library_stats(CID4)["documents"]
bot.posts.clear()
upload(bot, CID4, "dup.txt", TXT)
n2 = bot._library_stats(CID4)["documents"]
check("re-uploading the identical file keeps the document count at 1",
      n1 == 1 and n2 == 1,
      f"n1={n1} n2={n2} titles={[d['title'] for d in bot._library_stats(CID4)['docs']]}")

# a DIFFERENT file with the same name replaces, not duplicates, and the new
# content is what gets retrieved
bot.posts.clear()
upload(bot, CID4, "dup.txt", b"brand new different content entirely here")
st = bot._library_stats(CID4)
check("re-upload with new content still yields exactly one document",
      st["documents"] == 1, st)


# ── 5. My documents list: truncation note + no blind 4096 slice ────────────
print("=" * 70); print("5. long library list is noted/split, not silently cut")
bot2 = make_bot(); bot2._backend = T.InMemoryBackend()
CID2 = 881002
fresh(bot2, CID2)
lib = bot2._open_library(CID2)
stage = tempfile.mkdtemp(prefix="libfix_docs_")
paths = []
for i in range(60):
    p = pathlib.Path(stage) / (("verylongdocumentname" * 6) + f"_{i}.txt")
    p.write_text("hello world " * 50, encoding="utf-8")
    paths.append(str(p))
lib.build(paths)
lib.close()
check("60 documents actually indexed", bot2._library_stats(CID2)["documents"] == 60)

bot2.posts.clear()
bot2._send_library_list(CID2, bot2._get_session(CID2))
whole = "\n".join(sent(bot2))
shown_bullets = whole.count("• <b>")
check("every message stays under the hard Telegram cap",
      all(len(t) <= T._MAX_TEXT for t in sent(bot2)),
      [len(t) for t in sent(bot2)])
# The 40-item display cap is untouched by this fix (only its silence was the
# bug) — 60 documents still show 40 bullets, but now with an explicit
# "showing 40 of 60" note so the cut is not silent.
check("still shows the (unchanged) 40-item cap worth of bullets",
      shown_bullets == 40, shown_bullets)
check("a 'showing N of M' truncation note is present when the list is cut",
      ("40" in whole and "60" in whole and
       ("showing" in whole.lower() or "показано" in whole.lower())),
      whole[-200:])
check("no message text ends inside an unclosed HTML tag",
      all(t.rfind("<") <= t.rfind(">") for t in sent(bot2)),
      [t[-40:] for t in sent(bot2)])


# ── 6. per-file passage count, not the whole library's ─────────────────────
print("=" * 70); print("6. lib_indexed reports THIS file's passages")
bot = make_bot(); bot._backend = T.InMemoryBackend(); fresh(bot, CID)
big = ("A distinct sentence about topic one. " * 400).encode()
upload(bot, CID, "one.txt", big)
first_msg = last(bot)
one_chunks = next(d["chunks"] for d in bot._library_stats(CID)["docs"] if d["title"] == "one.txt")
check("one.txt: reported count matches its own real chunk count",
      f"({one_chunks} passages)" in first_msg or f"фрагментов: {one_chunks}" in first_msg,
      (first_msg, one_chunks))

bot.posts.clear()
upload(bot, CID, "two.txt", b"three word note")
second_msg = last(bot)
two_chunks = next(d["chunks"] for d in bot._library_stats(CID)["docs"] if d["title"] == "two.txt")
check("two.txt: reported count is its OWN chunks, not one.txt's total",
      (f"({two_chunks} passages)" in second_msg or f"фрагментов: {two_chunks}" in second_msg)
      and two_chunks < one_chunks,
      (second_msg, two_chunks, one_chunks))


# ── library.Library.build: stable source_id override ────────────────────────
print("=" * 70); print("7. library.py Library.build source_ids/titles override")
import library
DB = str(pathlib.Path(tempfile.mkdtemp(prefix="libfix_lib_")) / "t.db")
lib3 = library.Library(DB)
stageA = tempfile.mkdtemp(prefix="libfix_stageA_")
stageB = tempfile.mkdtemp(prefix="libfix_stageB_")
pA = str(pathlib.Path(stageA) / "safe_name.txt")
pathlib.Path(pA).write_text("some file content " * 20, encoding="utf-8")
lib3.build([pA], source_ids={pA: "tg:1:orig name.txt"}, titles={pA: "orig name.txt"})
pB = str(pathlib.Path(stageB) / "safe_name.txt")   # different temp dir, same content
pathlib.Path(pB).write_text("some file content " * 20, encoding="utf-8")
lib3.build([pB], source_ids={pB: "tg:1:orig name.txt"}, titles={pB: "orig name.txt"})
docs = lib3.documents()
check("same stable source_id across two different staged paths dedupes to one doc",
      len(docs) == 1, docs)
check("title override is honoured, not the staged basename",
      docs[0]["title"] == "orig name.txt", docs)
lib3.close()


print("=" * 70)
print(f"OK={OK} BAD={BAD}")
if BAD:
    sys.exit(1)
