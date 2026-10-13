"""The model's read of a picture's LOOK (imaging/ideogram_layout._look), stubbed
for the suites. LOOKS is the fixture table; bench/look_live.py runs the same
phrases on the real model."""
import types

import _sweep_stub
from fakes import reads

LOOKS = {
    "a realistic photo of a tank": "photo", "photorealistic portrait": "photo",
    "нарисуй реалистичное фото кота": "photo", "a hyperrealistic DSLR shot": "photo",
    "make it lifelike": "photo",
    "a cartoon cat": "cartoon", "anime girl in the rain": "anime",
    "an oil painting of a harbour": "oil painting", "мультяшный кот": "cartoon",
    "a watercolor sketch": "watercolour", "a realistic cartoon": "cartoon",
    "нарисуй машину": "none", "нарисуй машину во дворе": "none",
    "draw a cat sitting on a fence": "none", "картинка: закат над городом": "none",
    "a police car in a narrow yard": "none",
    "нарисуй карикатуру на полицейских": "caricature", "сделай шарж на кота": "caricature",
    "нарисуй мультяшную машину": "cartoon", "аниме девушка": "anime",
    "нарисуй акварелью мост": "watercolour", "комикс про кота": "comic strip",
    "On the wall behind the sofa hangs a painting depicting a calm sea.": "none",
    "a living room with a painting of the sea above the sofa": "none",
    "a boy holding a drawing of a dog": "none", "a desk with a sketch and a pencil": "none",
    "a girl reading a comic on the sofa": "none",
    "a realistic photo of a wall with a framed painting": "photo",
    "draw a painting of a harbour": "painting", "a painting of a cat": "painting",
    "make it look like a sketch": "sketch", "нарисуй кота в стиле живописи": "oil painting",
    "children's drawing style, a dog": "illustration",
    "нарисуй карикатуру, реалистично прорисованную": "caricature",
    "сделай шарж, но фотореалистичный": "caricature",
    "комикс в реалистичном стиле": "comic strip",
    "акварельный рисунок, как в жизни": "watercolour",
    "a realistic photo of a blue elephant on a beach": "photo",
    "draw a blue elephant on a beach": "none",
}

def pick(q, t):
    if "Which LOOK" in q:
        return LOOKS.get(t.strip(), "none")
    return _sweep_stub.pick(q, t)


READS = reads(yes=_sweep_stub.yes, pick=pick)


def reads_ctx(**more):
    """A context carrying the look table and the sweep table, plus `more` attributes."""
    return types.SimpleNamespace(**{**READS, **more})
