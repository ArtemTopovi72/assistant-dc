"""Editing a deck keeps its existing pictures when their slides survive."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services"))
import slides


def test_edit_reuses_existing_picture_without_search(tmp_path, monkeypatch):
    picture = tmp_path / "resistor.jpg"
    picture.write_bytes(b"existing-picture")
    previous = {"title": "Electronics", "slides": [{"heading": "Resistor", "image": "resistor"}],
                "_image_assets": [{"heading": "Resistor", "image": "resistor",
                                   "path": str(picture)}]}
    edited = {"title": "Electronics", "slides": [{"heading": "Resistor", "image": "resistor"}]}
    seen = {}
    monkeypatch.setattr(slides, "plan_deck", lambda *a, **kw: edited)
    monkeypatch.setattr(slides, "_drop_mismatched_images", lambda *a: None)
    monkeypatch.setattr(slides, "build_pptx", lambda deck, path, images, **kw:
                        seen.update(images) or path)
    monkeypatch.setattr(slides, "pptx_audit", lambda *a: {"empty": [], "overflow": [],
                        "offslide": [], "overlaps": [], "charts": 0})
    calls = []
    result = slides.make_presentation(None, "move images", str(tmp_path), previous=previous,
                                      image_fetcher=lambda q: calls.append(q) or None)
    assert result["path"] and seen[0] == str(picture) and calls == []
    assert result["deck"]["_image_assets"][0]["path"] == str(picture)


def test_content_diagram_is_not_cropped(tmp_path):
    from PIL import Image
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    diagram = tmp_path / "wide.png"
    Image.new("RGB", (1600, 450), "white").save(diagram)
    path = slides.build_pptx({"title": "Circuit", "slides": [
        {"heading": "Resistor", "bullets": ["It limits current"], "image": "diagram"}]},
        str(tmp_path / "deck.pptx"), {0: str(diagram)})
    pictures = [shape for shape in Presentation(path).slides[1].shapes
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    assert len(pictures) == 1
    assert all(getattr(pictures[0], side) == 0 for side in
               ("crop_left", "crop_right", "crop_top", "crop_bottom"))
