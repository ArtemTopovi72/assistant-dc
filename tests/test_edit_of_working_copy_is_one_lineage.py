"""An edit made from the working copy of an upload is a VERSION of that upload.

Live 2026-09-13 (journey 3): the user's photo is registered as
images/tg_<ts>.jpg, the graph edits a byte-identical _working_input_<ts>.jpg
and reports that path as image_derived_from. The path lookup found nothing,
the edited dress was registered as a second root, and "теперь убери все
надписи с фона" got "Which picture? There are 2" instead of the edit.
"""
import os, sys, types, tempfile, shutil
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

tmp = tempfile.mkdtemp()
try:
    upload = os.path.join(tmp, "tg_1.jpg"); open(upload, "wb").write(b"\xff\xd8photo-bytes" * 100)
    work = os.path.join(tmp, "_working_input_1.jpg"); shutil.copyfile(upload, work)
    other = os.path.join(tmp, "tg_2.jpg"); open(other, "wb").write(b"\xff\xd8other-bytes" * 100)
    edit = os.path.join(tmp, "edit.png"); open(edit, "wb").write(b"\x89PNG-edited" * 50)

    sess = types.SimpleNamespace(image_log=[])
    up_id = tg_bot._log_image(sess, upload, label="photo", src="user")
    tg_bot._log_image(sess, other, label="another", src="user")

    check("the working copy is not found by path", tg_bot._image_by_path(sess, work) is None)
    hit = tg_bot._image_by_content(sess, work)
    check("...but IS found by content", hit is not None and hit["id"] == up_id, hit)
    check("a different picture is not matched", tg_bot._image_by_content(sess, edit) is None)
    check("a missing path is handled", tg_bot._image_by_content(sess, os.path.join(tmp, "nope.jpg")) is None)
    check("an empty path is handled", tg_bot._image_by_content(sess, "") is None)

    # The lineage judgement that decides between "the picture" and "which one?"
    sess2 = types.SimpleNamespace(image_log=[])
    up2 = tg_bot._log_image(sess2, upload, label="photo", src="user")
    parent = (tg_bot._image_by_path(sess2, work) or tg_bot._image_by_content(sess2, work) or {}).get("id", "")
    tg_bot._log_image(sess2, edit, label="black dress", src="bot", parent=parent)
    check("the edit is registered as a child of the upload", parent == up2)
    check("upload + its edit is one lineage -> no 'which picture?'", tg_bot._one_lineage(tg_bot._live_images(sess2)))

    # The delivery layer uses the content fallback.
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_bot.__file__))), "bot/tg_tasks.py"), encoding="utf-8").read()
    check("tg_tasks falls back to the content match", "tg_bot._image_by_content(sess, _src)" in src)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
