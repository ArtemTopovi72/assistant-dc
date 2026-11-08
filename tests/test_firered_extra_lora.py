"""Klein-compatible task LoRAs stack after the edit LoRA and do not leak."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
from PIL import Image
import image as I
import klein

seen = []
from fakes import deps_ctx
CTX = deps_ctx(_upload_image_to_comfy=lambda p, u: "x.png", annotate_edit=lambda *a: None,
               _submit_and_poll=lambda ctx, wf, **k: seen.append(wf) or "out.png")
if True:
    p = os.path.join(tempfile.mkdtemp(), "s.png"); Image.new("RGB", (64, 64)).save(p)
    I.edit_image_with_firered(CTX, p, "remove the cup")
    assert "21" not in seen[-1] and seen[-1][klein.N_SHIFT]["inputs"]["model"] == [klein.N_LORA, 0]
    with I.firered_extra_lora("klein/task.safetensors", 0.8):
        I.edit_image_with_firered(CTX, p, "remove the cup")
    wf = seen[-1]
    assert wf["21"]["inputs"]["lora_name"] == "klein/task.safetensors"
    assert wf["21"]["inputs"]["model"] == [klein.N_LORA, 0] and wf[klein.N_SHIFT]["inputs"]["model"] == ["21", 0]
    with I.firered_extra_lora("remover/r.safetensors", 0.8):
        I.edit_image_with_firered(CTX, p, "remove the cup")
    assert "21" not in seen[-1], "incompatible LoRA must not be loaded"
    I.edit_image_with_firered(CTX, p, "remove the cup")
    assert "21" not in seen[-1], "the LoRA leaked past the block"
    print("4/4 ok")
