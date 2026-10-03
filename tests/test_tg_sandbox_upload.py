"""Uploading a NON-INDEXABLE file to a user who has a sandbox.

Found by driving the live bot. The assistant had just asked for a specific
.jar ("skip morevillagers-*.jar into that folder and I will pull the block ids
out of its registry"). The user sent exactly that file, and the bot answered:

    morevillagers-neoforge-1.21.1-6.0.0.jar is not a document I can add to
    your library (supported: .epub, .log, .markdown, .md, .pdf, ...). If you
    meant to attach it to a message, send it along with your question.

Every clause of which is wrong for this user. The file HAD been accepted --
tg_resolve stores the raw bytes in the sandbox before it ever reaches the
indexer -- and "send it along with your question" is precisely what they did.
The library rejection is correct in itself; it is just answering a question
nobody asked, because _index_document was called unconditionally.

So the rule these checks pin is: the library's opinion of an extension is only
worth reporting when the library was the file's only destination.

Run: venv/Scripts/python.exe tests/test_tg_sandbox_upload.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

# MUST come before any bot or store is built: without it the suite writes into
# the operator's live tg_users.db.
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_sbxup_")
T.redirect_data_dir(_DATA_DIR)

import code_sandbox as CS
import sandbox_access as SA

# redirect_data_dir moves the user DB, but NOT the sandbox tree. Without this
# the uploads below land in the operator's real runtime/sandboxes.
CS.redirect_sandbox_base(tempfile.mkdtemp(prefix="tgtest_sbxroot_"))

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


CID = 999877
_SANDBOX_BASE = tempfile.mkdtemp(prefix="tgtest_sbxroot_")


def _bot(level):
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent, bot.pushed = [], []
    bot._send_text = lambda cid, text, **k: (bot.sent.append((cid, text)), 1)[1]
    bot._send_get_id = lambda cid, text, **k: (bot.sent.append((cid, text)), 1)[1]
    bot._activity.log = lambda *a, **k: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._dl_bytes = lambda fid: b"PK\x03\x04 pretend jar bytes"
    real_push = bot._backend.push
    bot._backend.push = lambda t: (bot.pushed.append(t), real_push(t))[0]

    # The user row has to exist before a level can be granted to it.
    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    SA.set_level(bot._user_store, CID, level)
    return bot


def _send_jar(bot, caption=""):
    bot._resolve_and_push(CID, [{"type": "document", "file_id": "j1",
                                 "filename": "morevillagers-neoforge-1.21.1-6.0.0.jar",
                                 "caption": caption}])


def _texts(bot):
    return " || ".join(t for _, t in bot.sent)


def test_jar_to_a_user_with_a_sandbox_is_not_refused():
    orig_base = CS.SANDBOX_BASE if hasattr(CS, "SANDBOX_BASE") else None
    bot = _bot(SA.FILES)
    _send_jar(bot, "вытащи id блоков")
    body = _texts(bot)
    check("no library rejection for a sandboxed upload",
          "library" not in body.lower() and "библиотек" not in body.lower(), body)
    check("the turn still reaches the agent", len(bot.pushed) >= 1, bot.pushed)
    pushed = " ".join(str(t) for t in bot.pushed)
    check("the model is told where the file landed",
          ".jar" in pushed and "working folder" in pushed, pushed[:300])
    assert orig_base is None or True


def test_jar_without_a_sandbox_is_still_explained():
    """The rejection is right when the library really was the only destination."""
    bot = _bot(SA.OFF)
    _send_jar(bot, "что это?")
    body = _texts(bot)
    # With a caption the turn goes to the model, which is told plainly that the
    # file could not be read and why (tg_resolve scaffold, 2026-09-28); without
    # one the library's own rejection is the answer.
    pushed = " ".join(str(t) for t in getattr(bot, "pushed", []))
    check("a user with no sandbox still gets the explanation",
          "библиотек" in body.lower() or "library" in body.lower()
          or "not supported" in pushed, (body, pushed[:300]))


def test_a_stale_draw_prefix_cannot_seize_a_bare_upload():
    """The second defect from the same live session: 🎨 Рисую картинку…

    A .jar sent with NO caption makes a turn whose entire text is our own
    "[The file ... is now in your working folder]" note. A pending draw prefix
    left over from the Creativity menu was glued to the front of that, so
    _classify_task read it as an image request and the bot answered a file
    upload by generating a picture. A prompt prefix is an instruction about the
    user's next WORDS; with none present it must wait, not take the scaffolding.
    """
    bot = _bot(SA.FILES)
    sess = bot._store.get(CID)
    sess.pending_prefix = "Нарисуй: "
    bot._store.put(sess)

    _send_jar(bot, "")                       # no caption -- scaffolding only
    pushed = " ".join(str(t) for t in bot.pushed)
    check("the prefix did not attach to the scaffolding",
          "Нарисуй" not in pushed, pushed[:300])
    check("the turn is not classified as an image",
          T._classify_task(pushed) != T.KIND_IMAGE, T._classify_task(pushed))
    # no words, no turn: the user is told where the file went (owner 10-03)
    check("the user is told the file is in the sandbox",
          any("📁" in t for _, t in bot.sent), bot.sent[-2:])

    # ...and it still works the moment the user actually says something.
    bot.pushed.clear()
    bot._resolve_and_push(CID, [{"type": "text", "text": "кота"}])
    pushed2 = " ".join(str(t) for t in bot.pushed)
    check("a real message still consumes the prefix", "Нарисуй" in pushed2,
          pushed2[:200])


def test_indexing_is_by_the_button_only():
    """owner 10-03: «индексировать только по кнопке, остальное читать и класть в
    песочницу и говорить» -- crash logs got «📚 Индексирую…» and stalled."""
    bot = _bot(SA.FILES)
    seen = []
    bot._index_document = lambda cid, sess, path, name, **kw: seen.append(name)
    bot._dl_bytes = lambda fid: b"Traceback (most recent call last):\n  boom\n"
    for i, name in enumerate(("crash.log", "notes.txt", "guide.pdf")):
        bot._resolve_and_push(CID, [{"type": "document", "file_id": f"l{i}", "filename": name}])
    check("outside 📚 Documents nothing is indexed", seen == [], seen)
    said = " ".join(t for _, t in bot.sent)
    check("the user is told the file is in the sandbox", "crash.log" in said and "📁" in said, said[-300:])
    sess = bot._get_session(CID); sess.menu = "library"; bot._store.put(sess)
    bot._resolve_and_push(CID, [{"type": "document", "file_id": "p9", "filename": "book.pdf"}])
    check("inside 📚 Documents a file is indexed", seen == ["book.pdf"], seen)
    # owner 10-03: «у вкладки документы приоритет» -- even with a question
    bot._resolve_and_push(CID, [{"type": "text", "text": "в чём ошибка?"},
                                {"type": "document", "file_id": "p8", "filename": "crash.log"}])
    check("in 📚 Documents a file is indexed even with a question",
          seen == ["book.pdf", "crash.log"], seen)


def test_files_alone_run_no_turn_and_the_next_question_is_about_them():
    """live 10-03: forwarded logs alone -> «я всё изучил и готов» without opening
    them; then «В ЧЕМ ТАМ ОШИБКА?» critiqued the last picture"""
    bot = _bot(SA.FILES)
    bot._dl_bytes = lambda fid: b"Traceback (most recent call last):\n  boom\n"
    bot._resolve_and_push(CID, [{"type": "document", "file_id": "z1", "filename": "app.log"}])
    check("files with no words run no agent turn", bot.pushed == [], bot.pushed)
    bot._resolve_and_push(CID, [{"type": "text", "text": "в чём там ошибка?"}])
    turn = bot.pushed[-1].user_text if bot.pushed else ""
    check("the next question is pointed at the file", "app.log" in turn and "read_file" in turn, turn[-300:])


def test_a_batch_is_one_material_and_indexing_never_opens_documents():
    """owner 10-03: «он сам открыл вкладку документов» and «ответил по одному,
    хотя я заслал пачкой и ожидал комплексный ответ»"""
    bot = _bot(SA.FILES)
    bot._dl_bytes = lambda fid: b"Traceback (most recent call last):\n  boom\n"
    sess = bot._get_session(CID); sess.menu = ""; bot._store.put(sess)
    # a PDF without a sandbox-worthy question goes nowhere near the menu
    bot._resolve_and_push(CID, [{"type": "text", "text": "в чём ошибка?"}]
                          + [{"type": "document", "file_id": f"b{i}", "filename": n}
                             for i, n in enumerate(("a.log", "b.log", "c.log"))])
    turn = bot.pushed[-1].user_text if bot.pushed else ""
    check("one note names all three files", all(n in turn for n in ("a.log", "b.log", "c.log"))
          and "3 files" in turn and turn.count("[The file '") == 1, turn[-400:])
    # no sandbox -> the library indexes it; that must not switch the menu
    b2 = _bot(SA.OFF)
    b2._dl_bytes = lambda fid: b"some notes worth keeping\n" * 20
    s2 = b2._get_session(CID); s2.menu = ""; b2._store.put(s2)
    b2._resolve_and_push(CID, [{"type": "document", "file_id": "n1", "filename": "notes.txt"}])
    check("indexing did happen", any("notes.txt" in t for _, t in b2.sent), b2.sent[-3:])
    check("the menu was not switched to 📚 Documents", b2._get_session(CID).menu != "library",
          b2._get_session(CID).menu)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception:
            failed += 1
            import traceback; traceback.print_exc()
    passed = sum(1 for _, c in RESULTS if c)
    print("\n%d/%d functions, %d/%d checks passed"
          % (len(fns) - failed, len(fns), passed, len(RESULTS)))
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
