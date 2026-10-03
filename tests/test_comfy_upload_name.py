"""Uploads to ComfyUI get a name of their own, not the bare basename.

Bug: both upload paths sent os.path.basename(path) with overwrite=true. Every
combined contact sheet is video_<id>/sheet.jpg, so two chats animating theirs
at the same time both rendered from whichever upload landed last.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import comfy_client as C  # noqa: E402


def _two_sheets(tmp_path):
    a, b = tmp_path / "video_a" / "sheet.jpg", tmp_path / "video_b" / "sheet.jpg"
    for p, data in ((a, b"A"), (b, b"B")):
        p.parent.mkdir()
        p.write_bytes(data)
    return str(a), str(b)


def test_same_basename_different_files_get_different_names(tmp_path):
    a, b = _two_sheets(tmp_path)
    na, nb = C.upload_name(a), C.upload_name(b)
    assert na != nb
    assert na.startswith("sheet_") and na.endswith(".jpg")
    assert C.upload_name(a) == na               # stable for the same file


def test_both_upload_paths_send_the_unique_name(tmp_path, monkeypatch):
    import requests
    import video
    a, b = _two_sheets(tmp_path)
    sent = []

    class R:
        status_code = 200

        def json(self):
            return {}

    def post(url, files=None, **kw):
        sent.append(files["image"][0])
        return R()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(C.requests, "post", post, raising=False)
    names = [C._upload_image_to_comfy(a, "http://x"), C._upload_image_to_comfy(b, "http://x"),
             video._upload(a, "image"), video._upload(b, "image")]
    assert sent == [C.upload_name(a), C.upload_name(b)] * 2
    assert names == sent
