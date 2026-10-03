"""All of an Ozon card's photos go to vision as ONE sheet, one call (owner 10-03)."""
import io, os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("agent", "imaging", "research", "core"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ["F5_TEST_RUN"] = "1"
from PIL import Image
import tool_ozon_handlers as H


def _jpg():
    b = io.BytesIO(); Image.new("RGB", (50, 50), (200, 0, 0)).save(b, "JPEG"); return b.getvalue()


def test_one_vision_call(monkeypatch):
    import dr_urls, llm
    monkeypatch.setattr(dr_urls, "safe_get", lambda url, **k: types.SimpleNamespace(
        content=_jpg(), raise_for_status=lambda: None))
    calls = []
    monkeypatch.setattr(llm, "analyze_image_with_llm",
                        lambda **k: calls.append(k) or "tile 1: a red mug")
    ctx = types.SimpleNamespace(set_stage=lambda s: None)
    out = H._look_at_photos(ctx, {"name": "mug", "images": [f"https://x/{i}.jpg" for i in range(5)]})
    assert len(calls) == 1 and calls[0]["image_path"], calls
    assert "red mug" in out
    # the downloaded photos and the sheet do not stay behind in %TEMP%
    assert not os.path.exists(os.path.dirname(calls[0]["image_path"]))


if __name__ == "__main__":
    import pytest; sys.exit(pytest.main([__file__, "-q"]))
