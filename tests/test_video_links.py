"""A video link is downloaded and watched, not read as a page (live 10-03: a
YouTube Shorts link got «я не умею смотреть видео по ссылкам»)."""
import os, sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bot"))
import tg_links as L


def test_video_url_hosts():
    for u in ("https://youtube.com/shorts/5uI-Dae7-qs", "https://www.youtube.com/watch?v=x",
              "https://youtu.be/abc", "https://vt.tiktok.com/ZS/", "https://vk.com/clip-1_2",
              "https://rutube.ru/video/abc/", "https://x.com/user/status/123"):
        assert L.video_url("глянь " + u) == u, u
    for u in ("https://ru.wikipedia.org/wiki/Омлет", "https://youtube.com/@channel"):
        assert L.video_url(u) == "", u


class _FakeYDL:
    def __init__(self, opts): self.opts = opts
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def extract_info(self, url, download=False):
        return {"duration": int(url.rsplit("=", 1)[1]), "title": "t", "description": " d "}
    def download(self, urls):
        Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"MP4")


def test_fetch_video_downloads_and_caps_length():
    sys.modules["yt_dlp"] = types.SimpleNamespace(YoutubeDL=_FakeYDL)
    try:
        r = L.fetch_video("https://youtu.be/v?d=55")
        assert r == {"data": b"MP4", "seconds": 55, "title": "t", "description": "d"}, r
        r = L.fetch_video("https://youtu.be/v?d=4000")
        assert r == {"too_long": True, "seconds": 4000, "title": "t"}, r
    finally:
        del sys.modules["yt_dlp"]


if __name__ == "__main__":
    test_video_url_hosts()
    test_fetch_video_downloads_and_caps_length()
    print("ok")
