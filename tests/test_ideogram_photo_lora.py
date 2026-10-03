"""A photographic Ideogram render gets the Lenovo skin LoRA; a drawing or a
character LoRA does not (Ideogram's plastic skin, A/B 2026-10-02)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "imaging", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")
import ideogram as I
from ideogram_layout import build_caption, element

sent = []
def fake_run(ctx, workflow, *a, **k):
    sent.append(workflow)
    raise RuntimeError("stop")
I.comfy_client._submit_and_poll = fake_run

photo = build_caption("pier", [element("old fisherman", [100, 200, 900, 800])], photo="85mm portrait")
drawn = build_caption("pier", [element("old fisherman", [100, 200, 900, 800])], art_style="watercolor")
assert I.is_photo_caption(photo) and not I.is_photo_caption(drawn)

def loras(caption, **kw):
    sent.clear()
    try:
        I.generate(None, "x", caption=caption, **kw)
    except Exception:
        pass
    wf = sent[-1] if sent else {}
    return [n["inputs"]["lora_name"] for n in wf.values() if n.get("class_type") == "LoraLoaderModelOnly"]

assert I.IDEOGRAM_PHOTO_LORA in loras(photo), loras(photo)
assert I.IDEOGRAM_PHOTO_LORA not in loras(drawn), loras(drawn)
assert loras(photo, lora_name="char.safetensors") == ["char.safetensors"], loras(photo, lora_name="char.safetensors")
print("ok photo gets the skin LoRA, drawings and characters do not")
assert not I.is_photo_caption({"style_description": "photo, 85mm"}); print("ok a string style is not a crash")
