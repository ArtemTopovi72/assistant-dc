"""HOW a cloned voice says a line -- the one place to tune. audio.synth_single_segment calls pick() before synthesis and
finish() after; nothing else in the speech path knows about emotions.

pick(): chooses joy | anger | sad | neutral for the line (the model reads it; neutral unless it clearly carries the
        feeling) and swaps the voice's reference for the matching clip of its emotion pack (voice/emopack.py).
        Voices with no pack yet get one built in the background and speak plainly meanwhile.
finish(): a one-sentence question gets the Russian IK-3 contour (voice/prosody.py).
Explicit control: ctx.delivery_emotion ("" = plain, None = automatic), ctx.delivery_ik3 (False = off).
Off switch: EMOTION_DELIVERY=0.  Only cloned voices (ctx.custom_ref_wav) are touched; failures fall back to plain speech.
"""
import logging
import os

logger = logging.getLogger(__name__)
OPTIONS = ("neutral", "joy", "anger", "sad")
MIN_CHARS = 12                       # "Да." / "Хорошо." carry no emotion worth a model call
_asked = set()


def enabled() -> bool:
    return os.getenv("EMOTION_DELIVERY", "1") != "0"


def emotion_of(ctx, text: str) -> str:
    explicit = getattr(ctx, "delivery_emotion", None)
    if explicit is not None:
        return explicit if explicit in OPTIONS else ""
    if len((text or "").strip()) < MIN_CHARS:
        return ""
    import intent
    got = intent.ask_choice(
        "Which feeling does the speaker voice in this Russian line? Say neutral unless the line clearly shows joy "
        "(delight, excitement, laughter), anger (irritation, shouting, rebuke) or sad (grief, disappointment, longing). {text}",
        text, OPTIONS, default="neutral")
    return got if got != "neutral" else ""


def pick(ctx, ref_wav: str, ref_text: str, text: str):
    """-> (ref_wav, ref_text) for this line."""
    if not enabled() or not ref_wav:
        return ref_wav, ref_text
    try:
        import emopack
        if not emopack.has_pack(ref_wav):
            if ref_wav not in _asked:
                _asked.add(ref_wav)
                emopack.build_async(ref_wav, ctx)
            return ref_wav, ref_text
        emo = emotion_of(ctx, text)
        return emopack.ref_for(ref_wav, emo) or (ref_wav, ref_text)
    except Exception:                                            # noqa: BLE001 -- plain speech beats no speech
        logger.warning("delivery.pick failed", exc_info=True)
        return ref_wav, ref_text


def finish(ctx, wav: str, text: str) -> str:
    if not enabled() or getattr(ctx, "delivery_ik3", None) is False:
        return wav
    try:
        import prosody
        if getattr(ctx, "delivery_ik3", None) or prosody.one_question(text):
            return prosody.question_shape(ctx, wav, text)
    except Exception:                                            # noqa: BLE001
        logger.warning("delivery.finish failed", exc_info=True)
    return wav
