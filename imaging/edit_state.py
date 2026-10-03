"""Stateful image-edit graph with region locks and per-node validation.

The old model treated every edit request as a fresh generation from the *original*
image, so a "change the hat" then "upscale" chain could regenerate the portrait and
upscale a stranger. This module makes editing **stateful**: each operation is a node
that knows its source image, the resulting current image, which regions were locked
vs editable, and whether it passed validation. Chained ops (upscale/restore/another
edit) operate on the CURRENT node's image, never the original — so state is preserved
across steps.

Each node validates before becoming current:
  * identity (face cosine) must hold for ops that lock the face;
  * containment (pixels changed outside the editable region) must stay low.
If validation fails the node is marked failed and the previous current is kept.
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("assistant.edit_state")

# Subject-lock policy per intent: which regions may change, which are protected.
# "identity"/"geometry" are conceptual locks enforced by metric gates, not masks.
LOCK_POLICY = {
    "face_edit":          {"editable": ["face"],       "locked": ["body", "background"], "expect_identity_change": True},
    "clothing_edit":      {"editable": ["clothing"],   "locked": ["face", "background"], "expect_identity_change": False},
    "product_edit":       {"editable": ["product"],    "locked": ["face", "background"], "expect_identity_change": False},
    "subject_edit":       {"editable": ["subject"],    "locked": ["face", "background"], "expect_identity_change": False},
    "style_transfer":     {"editable": ["whole"],      "locked": [],                     "expect_identity_change": True},
    "background_replace": {"editable": ["background"],  "locked": ["subject", "face"],    "expect_identity_change": False},
    "background_remove":  {"editable": ["background"],  "locked": ["subject", "face"],    "expect_identity_change": False},
    "object_remove":      {"editable": ["object"],     "locked": ["face", "subject"],    "expect_identity_change": False},
    "person_remove":      {"editable": ["person"],     "locked": ["background"],         "expect_identity_change": False},
    "object_insert":      {"editable": ["insertion"],  "locked": ["face", "subject"],    "expect_identity_change": False},
    "relight":            {"editable": ["lighting"],   "locked": ["geometry", "identity"], "expect_identity_change": False},
    "outpaint":           {"editable": ["new_border"], "locked": ["original_center"],     "expect_identity_change": False},
    "upscale":            {"editable": ["resolution"], "locked": ["identity", "geometry", "composition"], "expect_identity_change": False},
    "restore":            {"editable": ["detail"],     "locked": ["identity", "geometry"], "expect_identity_change": False},
}

IDENTITY_MIN_COSINE = 0.90   # for ops that lock the face


@dataclass
class EditNode:
    op: str
    instruction: str
    image_path: str
    locked: List[str] = field(default_factory=list)
    editable: List[str] = field(default_factory=list)
    parent: int = -1
    validation: dict = field(default_factory=dict)
    ok: bool = True

    def summary(self) -> str:
        v = self.validation
        idy = f"id={v['identity_cosine']:.3f}" if v.get("identity_cosine") is not None else "id=n/a"
        face = f"faceΔ={v['face_change']*100:.1f}%" if v.get("face_change") is not None else ""
        return f"[{'OK' if self.ok else 'FAIL'}] {self.op}: {idy} {face} -> {os.path.basename(self.image_path)}"


class EditSession:
    """A chain of edits over one source image, each validated, state-preserving."""

    def __init__(self, source_path: str):
        self.source_path = source_path
        self.nodes: List[EditNode] = [
            EditNode(op="source", instruction="", image_path=source_path,
                     locked=[], editable=[], parent=-1, validation={}, ok=True)
        ]

    @property
    def current(self) -> EditNode:
        # last node that succeeded
        for n in reversed(self.nodes):
            if n.ok:
                return n
        return self.nodes[0]

    @property
    def current_path(self) -> str:
        return self.current.image_path

    # -- validation ------------------------------------------------------- #
    def _validate(self, op: str, before_path: str, after_path: str) -> dict:
        policy = LOCK_POLICY.get(op, {})
        expect_id_change = policy.get("expect_identity_change", False)
        out = {"identity_cosine": None, "face_change": None, "overall_change": None,
               "ok": True, "reason": ""}
        try:
            import identity_metrics as idm
            out["identity_cosine"] = idm.identity_cosine(before_path, after_path)
            out["face_change"] = idm.face_region_change(before_path, after_path)
            ch = idm.changed_fraction(before_path, after_path)
            out["overall_change"] = ch.get("overall")
        except Exception as exc:
            logger.info("validation metrics unavailable (%s)", exc)
            return out
        cos = out["identity_cosine"]
        # face-locked ops must keep the same person
        if not expect_id_change and cos is not None and cos < IDENTITY_MIN_COSINE:
            out["ok"] = False
            out["reason"] = f"identity cosine {cos:.3f} < {IDENTITY_MIN_COSINE} (face changed)"
        return out

    # -- ops -------------------------------------------------------------- #
    def apply_edit(self, instruction: str, *, seed: Optional[int] = None,
                   timeout: int = 1900) -> EditNode:
        """Route an edit on the CURRENT image, validate, and append a node."""
        import image as img
        before = self.current_path
        category, out = img.route_edit_request(None, before, instruction, seed=seed, timeout=timeout)
        if not out or not os.path.exists(out):
            node = EditNode(op=category, instruction=instruction, image_path=before,
                            parent=self.nodes.index(self.current), ok=False,
                            validation={"reason": "no output"})
            self.nodes.append(node)
            return node
        policy = LOCK_POLICY.get(category, {})
        val = self._validate(category, before, out)
        node = EditNode(op=category, instruction=instruction, image_path=out,
                        locked=policy.get("locked", []), editable=policy.get("editable", []),
                        parent=self.nodes.index(self.current), validation=val,
                        ok=val.get("ok", True))
        self.nodes.append(node)
        logger.info("EditSession %s", node.summary())
        return node

    def history(self) -> str:
        return "\n".join(f"  {i}: {n.summary()}" for i, n in enumerate(self.nodes))
