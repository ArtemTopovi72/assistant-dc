"""Generated media gets one useful title without making delivery depend on the model."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent"))
import nice_names as names


def test_generated_media_title_is_reused_for_the_same_request(tmp_path, monkeypatch):
    monkeypatch.setattr(names, "_ROOT", str(tmp_path / "archive"))
    calls = []
    monkeypatch.setattr(names, "_suggest_title", lambda request, path: calls.append(request) or "Советский трактор")
    names.TITLES[123456] = "Сделай советский норм трактор"
    for suffix in ("1", "2"):
        source = tmp_path / f"ComfyUI_{suffix}0000.png"
        source.write_bytes(b"image")
        archived = names.named_path(123456, str(source))
        assert os.path.basename(archived).startswith("Советский трактор")
    names.TITLES[123456] = "Сделай советский норм трактор"  # a repeat turn
    source = tmp_path / "ComfyUI_30000.png"
    source.write_bytes(b"image")
    assert os.path.basename(names.named_path(123456, str(source))).startswith("Советский трактор")
    assert calls == ["Сделай советский норм трактор"]


def test_failed_title_falls_back_and_user_files_keep_their_names(tmp_path, monkeypatch):
    monkeypatch.setattr(names, "_ROOT", str(tmp_path / "archive"))
    calls = []
    monkeypatch.setattr(names, "_suggest_title", lambda request, path: calls.append(path) or "")
    names.TITLES[123457] = "Сделай советский норм трактор"
    generated = tmp_path / "ComfyUI_00115_.png"
    generated.write_bytes(b"image")
    assert os.path.basename(names.named_path(123457, str(generated))) == "советский трактор.png"
    original = tmp_path / "Даша.png"
    original.write_bytes(b"image")
    assert os.path.basename(names.named_path(123457, str(original))) == "Даша.png"
    assert calls == [str(generated)]
