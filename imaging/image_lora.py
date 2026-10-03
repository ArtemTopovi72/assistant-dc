"""Hang a character LoRA on a ComfyUI workflow.

The committed workflow has no LoRA node at all, and it has TWO sampling stages
(base and refine) each fed by its own UNETLoader. Wiring the adapter into only
the first stage is the trap here: the refine pass would then re-render the
face with the plain base model and quietly undo the likeness, which looks like
"the LoRA is weak" rather than "the LoRA was skipped".

So injection is done by rewiring every consumer of a UNETLoader, not by
patching one named node id -- node ids in that file are stage-prefixed strings
("57:11", "96:70") and are not something to hardcode against.

train_text_encoder is false in our training config, so the adapter is
model-only; LoraLoaderModelOnly is the matching node and leaves CLIP untouched.
"""
from __future__ import annotations

import copy

_MODEL_LOADERS = ("UNETLoader", "CheckpointLoaderSimple")


def lora_node_id(base: str) -> str:
    return "lora_" + str(base)


def inject_lora(graph: dict, lora_name: str, strength: float = 0.9) -> dict:
    """Return a copy of `graph` with `lora_name` applied to every model path.

    Raises ValueError when the graph has no model loader to hang it on --
    silently returning the untouched graph would render a stranger and call it
    the character.
    """
    if not lora_name:
        raise ValueError("no LoRA file given")
    g = copy.deepcopy(graph)
    loaders = [nid for nid, n in g.items()
               if isinstance(n, dict) and n.get("class_type") in _MODEL_LOADERS]
    if not loaders:
        raise ValueError("workflow has no UNET/checkpoint loader to attach a LoRA to")

    added = {}
    for nid in loaders:
        new_id = lora_node_id(nid)
        added[new_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": lora_name,
                       "strength_model": float(strength),
                       "model": [nid, 0]},
        }

    # Rewire consumers BEFORE adding the new nodes, so the loader nodes we just
    # created do not get rewired to point at themselves.
    for nid, node in g.items():
        if not isinstance(node, dict):
            continue
        for key, val in list(node.get("inputs", {}).items()):
            if isinstance(val, list) and len(val) == 2 and str(val[0]) in loaders:
                node["inputs"][key] = [lora_node_id(val[0]), 0]
    g.update(added)
    return g


def strip_lora(graph: dict) -> dict:
    """Undo inject_lora. Used when a character is deselected mid-session."""
    g = copy.deepcopy(graph)
    nodes = {nid: n for nid, n in g.items()
             if isinstance(n, dict) and n.get("class_type") == "LoraLoaderModelOnly"}
    for nid, node in nodes.items():
        src = node.get("inputs", {}).get("model")
        for other in g.values():
            if not isinstance(other, dict):
                continue
            for key, val in list(other.get("inputs", {}).items()):
                if isinstance(val, list) and len(val) == 2 and val[0] == nid:
                    other["inputs"][key] = list(src)
        del g[nid]
    return g


def prompt_with_trigger(prompt: str, trigger: str) -> str:
    """Put the trigger word in front, once.

    An adapter attaches to its trigger; a prompt without it renders whoever the
    base model felt like. Appending a second copy when the user already typed
    it is not harmless either -- a repeated token gets extra weight.
    """
    p = (prompt or "").strip()
    t = (trigger or "").strip()
    if not t:
        return p
    if t.lower() in p.lower():
        return p
    return (t + ", " + p) if p else t
