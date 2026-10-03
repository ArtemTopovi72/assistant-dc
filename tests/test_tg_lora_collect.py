"""Admin-only photo intake for the style LoRA dataset, driven through the REAL bot.

Built for one real workflow: the operator's brother drops reference photos
(Frank Horrigan, NCR soldiers, the Master...) with captions into the chat, and
they need to land in runtime/lora_datasets/fallout_style/ exactly like a photo
dropped through the desktop tab would. The interesting failure modes are all
about the mode NOT leaking: an ordinary user must never be able to arm it, a
Stop/cancel must actually turn it off (not just look like it did on screen),
and an unrelated stray message while armed must not get swallowed as training
data.

Run: venv/Scripts/python.exe tests/test_tg_lora_collect.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_loracollect_")
T.redirect_data_dir(_DATA_DIR)

import characters as C
import lora_training as LT
_STORE_DIR = tempfile.mkdtemp(prefix="tgtest_loracollect_store_")
C.redirect_store(os.path.join(_STORE_DIR, "characters.json"))
_DATASET_ROOT = tempfile.mkdtemp(prefix="tgtest_loracollect_ds_")
LT.DATASET_ROOT = __import__("pathlib").Path(_DATASET_ROOT)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


CID = 999879
_next_fid = [0]


def _bot(is_admin=True):
    for slug in [c["slug"] for c in C.list_characters()]:
        C.delete(slug)
    import shutil
    ds = LT.DATASET_ROOT / "fallout_style"
    if ds.is_dir():
        shutil.rmtree(ds)
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **k: bot.sent.append(text) or 1
    bot._send_get_id = bot._send_text
    bot._activity.log = lambda *a, **k: None
    bot._api_post = lambda *a, **k: {}
    _next_fid[0] += 1
    bot._dl_bytes = lambda fid: (b"" if fid == "BAD" else b"jpegbytes" + fid.encode())
    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    bot._sessions.pop(CID, None)
    sess = bot._get_session(CID)
    sess.reg_state = ""
    sess.is_admin = is_admin
    bot._store.put(sess)
    return bot


def _texts(bot):
    return " || ".join(bot.sent)


def _sess(bot):
    return bot._get_session(CID)


def _dataset_files():
    d = LT.DATASET_ROOT / "fallout_style"
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def test_a_non_admin_cannot_arm_collection():
    bot = _bot(is_admin=False)
    bot._cb_start_lora_collect(CID)
    check("refused", "администратору" in _texts(bot), _texts(bot))
    check("mode not armed", _sess(bot).reg_state != "lora_collect")


def test_admin_arms_and_a_captioned_photo_is_filed():
    bot = _bot()
    bot._cb_start_lora_collect(CID)
    check("mode armed", _sess(bot).reg_state == "lora_collect")
    bot._resolve_and_push(CID, [{"type": "photo", "file_id": "f1",
                                 "caption": "Frank Horrigan, Enclave power armor"}])
    files = _dataset_files()
    check("one image + one caption saved", len(files) == 2, files)
    txts = [f for f in files if f.endswith(".txt")]
    if txts:
        text = (LT.DATASET_ROOT / "fallout_style" / txts[0]).read_text(encoding="utf-8")
        check("caption is verbatim, not stripped",
              text == "Frank Horrigan, Enclave power armor", text)
    check("registry has the style entry",
          C.get("fallout_style") is not None)


def test_an_uncaptioned_photo_still_gets_a_caption_file():
    """The trainer expects a .txt beside every image; a caption-less drop
    must not silently produce an orphan .jpg with nothing to pair it."""
    bot = _bot()
    bot._cb_start_lora_collect(CID)
    bot._resolve_and_push(CID, [{"type": "photo", "file_id": "f2", "caption": ""}])
    files = _dataset_files()
    check("no caption still saved a pair", len(files) == 2, files)


def test_an_album_shares_one_caption_across_all_its_photos():
    bot = _bot()
    bot._cb_start_lora_collect(CID)
    bot._resolve_and_push(CID, [{"type": "album",
                                 "file_ids": ["a1", "a2", "a3"],
                                 "caption": "NCR soldiers, three angles"}])
    files = _dataset_files()
    jpgs = [f for f in files if f.endswith(".jpg")]
    txts = [f for f in files if f.endswith(".txt")]
    check("all three photos saved", len(jpgs) == 3, files)
    check("each got its own caption file", len(txts) == 3, files)
    contents = {(LT.DATASET_ROOT / "fallout_style" / t).read_text(encoding="utf-8")
                for t in txts}
    check("all three share the album's one caption",
          contents == {"NCR soldiers, three angles"}, contents)


def test_a_failed_download_does_not_silently_lose_the_slot():
    bot = _bot()
    bot._cb_start_lora_collect(CID)
    bot._resolve_and_push(CID, [{"type": "photo", "file_id": "BAD", "caption": "x"}])
    check("failure is reported", "Не смог скачать" in _texts(bot), _texts(bot))
    check("no half-written pair", _dataset_files() == [])


def test_stop_actually_turns_collection_off():
    """The defect that matters: Stop must not just SHOW the main keyboard
    while collection quietly keeps eating every photo sent afterward."""
    bot = _bot()
    bot._cb_start_lora_collect(CID)
    bot._stop_and_report(CID, _sess(bot), "ru")
    check("reg_state cleared", _sess(bot).reg_state == "",
          _sess(bot).reg_state)
    bot._resolve_and_push(CID, [{"type": "photo", "file_id": "f9",
                                 "caption": "should not be collected"}])
    check("a photo sent after Stop is NOT filed as training data",
          _dataset_files() == [], _dataset_files())


def test_collection_mode_does_not_reach_the_agent():
    """A photo/album consumed by collection mode must never also become an
    ordinary picture message pushed onto the task queue."""
    bot = _bot()
    bot._cb_start_lora_collect(CID)
    pushed = []
    bot._backend.push = lambda t: pushed.append(t)
    bot._resolve_and_push(CID, [{"type": "photo", "file_id": "f10",
                                 "caption": "test"}])
    check("nothing was pushed to the agent queue", pushed == [], pushed)


def _main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as exc:
            check(fn.__name__ + " raised", False, exc)
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
