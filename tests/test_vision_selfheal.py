"""The vision self-test heals itself: a stray f16 projector beside the BF16
is moved aside, the model reloaded, and the picture asked again."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vision_selftest as V



def test_stray_f16_projector_is_moved_aside_only_beside_a_bf16(tmp_path):
    both = tmp_path / "a"; both.mkdir()
    (both / "mmproj-X-BF16.gguf").write_bytes(b"1")
    (both / "mmproj-X-f16.gguf").write_bytes(b"2")
    alone = tmp_path / "b"; alone.mkdir()
    (alone / "mmproj-Y-f16.gguf").write_bytes(b"3")
    moved = V.quarantine_f16_projectors(str(tmp_path))
    assert [os.path.basename(m) for m in moved] == ["mmproj-X-f16.gguf.broken-f16.bak"]
    assert (both / "mmproj-X-BF16.gguf").exists() and not (both / "mmproj-X-f16.gguf").exists()
    assert (alone / "mmproj-Y-f16.gguf").exists()         # its only projector
    (both / "mmproj-X-f16.gguf").write_bytes(b"2")        # it came back
    assert V.quarantine_f16_projectors(str(tmp_path))[0].endswith(".broken-f16.bak2")


def test_blind_answer_heals_once_and_asks_again(monkeypatch):
    answers = iter(["There is no animal in this picture, and the sofa is black.",
                    "Yes, a black cat on a green sofa."])
    calls = []
    monkeypatch.delenv("F5_TEST_RUN", raising=False)
    monkeypatch.setattr(V, "_ask", lambda ctx: next(answers))
    monkeypatch.setattr(V, "quarantine_f16_projectors", lambda *a: calls.append("q") or [])
    monkeypatch.setattr(V, "_reload_house_model", lambda ctx: calls.append("reload") or True)
    monkeypatch.setitem(V._HEALED, "done", False)
    assert V.run(object()) is True
    assert calls == ["q", "q", "reload"]
    assert V.run.__doc__ and V._HEALED["done"]
