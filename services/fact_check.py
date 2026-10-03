"""Is a claim supported by the notes it was written from? MiniCheck (Flan-T5-Large, MIT).

The best checker under 1B parameters on LLM-AggreFact (Tang et al., 2024): the
document and the claim go in, one forward pass, the probability of "supported"
comes out. The inference is the reference code's (Liyan06/MiniCheck), done here
directly on transformers, CPU: the chat model owns the card.

English only: a claim in another language is translated before it is checked.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger("assistant.fact_check")

MODEL = "lytang/MiniCheck-Flan-T5-Large"
CHUNK_WORDS = 400
_lock = threading.Lock()
_model = None
_failed = False


def _load():
    global _model, _failed
    with _lock:
        if _model is None and not _failed:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(MODEL)
                m = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
                _model = (tok, m)
            except Exception:
                _failed = True
                logger.warning("MiniCheck unavailable -- claims go unchecked", exc_info=True)
    return _model


def support(claims: List[str], notes: str) -> Optional[List[float]]:
    """P(supported) per claim, the best over the notes' chunks. None when it cannot run."""
    if os.getenv("F5_TEST_RUN") or not claims or not (notes or "").strip():
        return None
    got = _load()
    if not got:
        return None
    import torch
    tok, m = got
    words = notes.split()
    chunks = [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]
    out = []
    with _lock, torch.no_grad():
        for c in claims:
            x = tok(["predict: " + ch + tok.eos_token + c for ch in chunks],
                    return_tensors="pt", padding=True, truncation=True, max_length=2048)
            logits = m(**x, decoder_input_ids=torch.zeros((len(chunks), 1), dtype=torch.long)).logits
            out.append(float(logits[:, 0, [3, 209]].softmax(-1)[:, 1].max()))
    return out
