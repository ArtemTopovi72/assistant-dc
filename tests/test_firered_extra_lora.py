"""FireRed edits inside firered_extra_lora() stack the task LoRA after Lightning."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
from PIL import Image
import image as I

seen = []
_o_up, _o_sub = I._upload_image_to_comfy, I._submit_and_poll
I._upload_image_to_comfy = lambda p, u: "x.png"
I._submit_and_poll = lambda ctx, wf, **k: seen.append(wf) or "out.png"
try:
    p = os.path.join(tempfile.mkdtemp(), "s.png"); Image.new("RGB", (64, 64)).save(p)
    I.edit_image_with_firered(None, p, "remove the cup")
    assert "183x" not in seen[-1] and seen[-1]["172"]["inputs"]["model"] == ["183", 0]
    with I.firered_extra_lora("remover/r.safetensors", 0.8):
        I.edit_image_with_firered(None, p, "remove the cup")
    wf = seen[-1]
    assert wf["183x"]["inputs"]["lora_name"] == "remover/r.safetensors"
    assert wf["183x"]["inputs"]["model"] == ["183", 0] and wf["172"]["inputs"]["model"] == ["183x", 0]
    I.edit_image_with_firered(None, p, "remove the cup")
    assert "183x" not in seen[-1], "the LoRA leaked past the block"
    print("3/3 ok")
finally:
    I._upload_image_to_comfy, I._submit_and_poll = _o_up, _o_sub
