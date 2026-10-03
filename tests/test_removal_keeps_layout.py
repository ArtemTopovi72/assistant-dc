import image as I
import image_router as R


def test_removal_carries_layout(tmp_path, monkeypatch):
    src, out = tmp_path / "a.png", tmp_path / "b.png"
    src.write_bytes(b"x"); out.write_bytes(b"x")
    I.save_layout_for(str(src), "p", {"elements": [1]}, width=10, height=10, seed=3)
    monkeypatch.setattr(R, "_route_edit_request", lambda *a, **k: ("object_remove", str(out)))
    assert R.route_edit_request(None, str(src), "remove the dog") == ("object_remove", str(out))
    assert I.load_layout_for(str(out))["seed"] == 3


def test_repaint_does_not_carry(tmp_path, monkeypatch):
    src, out = tmp_path / "a.png", tmp_path / "b.png"
    src.write_bytes(b"x"); out.write_bytes(b"x")
    I.save_layout_for(str(src), "p", {"elements": [1]}, width=10, height=10)
    monkeypatch.setattr(R, "_route_edit_request", lambda *a, **k: ("style_transfer", str(out)))
    R.route_edit_request(None, str(src), "make it anime")
    assert I.load_layout_for(str(out)) is None


def test_draw_chart_keeps_generator():
    import tools, tool_retrieval as T
    names = lambda w: [T._name(s) for s in T.select_tools("", tools.TOOL_SCHEMAS, wants=w)]
    assert "generate_image" in names(["run_code", "generate_image"])   # «нарисуй график ... картинкой»
    assert "generate_image" not in names(["run_code"])                 # «построй питоном график»
