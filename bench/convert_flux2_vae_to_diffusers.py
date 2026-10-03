"""Rename the ComfyUI flux2 VAE into the diffusers key form the trainer wants.

ideogram4.py calls convert_diffusers_state_dict() unconditionally, and that
function maps diffusers -> native and RAISES on anything else. ComfyUI ships
the VAE already in native form (decoder.mid.attn_1.k.bias), so the trainer
rejects a file that is, in substance, exactly what it wants -- the attention
weights are even already 4-D, which is the only reshape the converter performs.

Rather than patch the vendored toolkit, this renames our keys into the
diffusers form, so the trainer's own converter turns them straight back. That
is a round trip, and a round trip is the point: every produced key is fed
through the TOOLKIT's own _rewrite_diffusers_key and must come back as the
native key we started from. A rule I got wrong therefore fails here, loudly,
instead of loading a subtly wrong VAE.

    venv/Scripts/python.exe bench/convert_flux2_vae_to_diffusers.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "ai-toolkit"))

from safetensors import safe_open
from safetensors.torch import save_file

from toolkit.models.v2.vae.flux2_kl import _NUM_RESOLUTIONS, _rewrite_diffusers_key

VAE = Path(r"E:\ideogram4-local\vae\diffusion_pytorch_model.safetensors")


def to_diffusers(key: str):
    """Native key -> diffusers key. None when the key needs no rename."""
    if key.startswith(("bn.", "encoder.conv_in.", "encoder.conv_out.",
                       "decoder.conv_in.", "decoder.conv_out.")):
        return key
    if key.startswith("encoder.quant_conv."):
        return key[len("encoder."):]
    if key.startswith("decoder.post_quant_conv."):
        return key[len("decoder."):]
    for side in ("encoder", "decoder"):
        if key == "%s.norm_out.weight" % side:
            return "%s.conv_norm_out.weight" % side
        if key == "%s.norm_out.bias" % side:
            return "%s.conv_norm_out.bias" % side

    m = re.match(r"^(encoder|decoder)\.mid\.block_(\d+)\.(.+)$", key)
    if m:
        side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        return "%s.mid_block.resnets.%d.%s" % (
            side, idx - 1, rest.replace("nin_shortcut", "conv_shortcut"))
    m = re.match(r"^(encoder|decoder)\.mid\.attn_1\.(.+)$", key)
    if m:
        side, rest = m.group(1), m.group(2)
        rest = (rest.replace("norm.", "group_norm.")
                    .replace("q.", "to_q.").replace("k.", "to_k.")
                    .replace("v.", "to_v.").replace("proj_out.", "to_out.0."))
        return "%s.mid_block.attentions.0.%s" % (side, rest)

    m = re.match(r"^encoder\.down\.(\d+)\.block\.(\d+)\.(.+)$", key)
    if m:
        return "encoder.down_blocks.%s.resnets.%s.%s" % (
            m.group(1), m.group(2), m.group(3).replace("nin_shortcut", "conv_shortcut"))
    m = re.match(r"^encoder\.down\.(\d+)\.downsample\.conv\.(.+)$", key)
    if m:
        return "encoder.down_blocks.%s.downsamplers.0.conv.%s" % (m.group(1), m.group(2))

    m = re.match(r"^decoder\.up\.(\d+)\.block\.(\d+)\.(.+)$", key)
    if m:
        native = int(m.group(1))
        return "decoder.up_blocks.%d.resnets.%s.%s" % (
            _NUM_RESOLUTIONS - 1 - native, m.group(2),
            m.group(3).replace("nin_shortcut", "conv_shortcut"))
    m = re.match(r"^decoder\.up\.(\d+)\.upsample\.conv\.(.+)$", key)
    if m:
        native = int(m.group(1))
        return "decoder.up_blocks.%d.upsamplers.0.conv.%s" % (
            _NUM_RESOLUTIONS - 1 - native, m.group(2))
    return None


def main() -> int:
    if not VAE.is_file():
        print("missing:", VAE)
        return 1
    tensors = {}
    with safe_open(str(VAE), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    print("%d tensors" % len(tensors))

    if any(k.startswith(("encoder.down_blocks.", "decoder.up_blocks.")) for k in tensors):
        print("already in diffusers form -- nothing to do")
        return 0

    out, bad = {}, []
    for k, t in tensors.items():
        d = to_diffusers(k)
        if d is None:
            bad.append((k, "no rule"))
            continue
        back = _rewrite_diffusers_key(d)
        if back != k:                       # the round trip IS the test
            bad.append((k, "%s -> %s" % (d, back)))
            continue
        out[d] = t
    if bad:
        print("FAILED to round-trip %d keys:" % len(bad))
        for k, why in bad[:10]:
            print("   %-50s %s" % (k, why))
        return 1

    print("all %d keys round-trip through the toolkit's own converter" % len(out))
    save_file(out, str(VAE))
    print("rewrote", VAE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
