"""The model's narrow reads (intent.ask_yes / ask_choice) for the 10-02 sweep,
stubbed for the suites as fixture tables: each fixture phrase -> what the
model answers. bench/intent_sweep3_live.py runs these questions on the model.

Importing installs the stubs; an earlier stub stays reachable for the
questions this table does not know."""
import intent

REMOVES = {   # (instruction, lettering) -> does the instruction remove it
    ("убери дату", "12 октября"): True, ("удали заголовок", "РОК НОЧЬ"): True,
    ("remove the date", "12 October"): True, ("убери собаку", "12 октября"): False,
    ("Вику убери", "Вика"): True, ("телефон сделай крупнее, а адрес убери", "ул. Ленина 5"): True,
    ("убери второй шаг", "засыпать заварку"): True, ("убери второй шаг", "нагреть воду"): False,
    ("убери второй шаг", "2. Засыпать заварку"): True,
    ("убери среднюю банку", "CHERRY"): True,   # the jar goes with its label
}
TRANSFORM = {"rotate the image 90 degrees": "rotate_right", "flip it horizontally": "mirror",
             "crop to the face": "crop", "flip it vertically": "flip"}
PLACE = {"подпись снизу": "bottom", "в небо": "top", "по центру": "middle"}
KIND = {"поменяй фон на картинке": "image"}
ORIENT = {"a wide landscape banner": "landscape", "vertical portrait phone wallpaper": "portrait",
          "a single person standing, headshot": "portrait", "a city street crowd panorama": "landscape",
          "abstract": "unknown", "single person headshot": "portrait", "city panorama": "landscape",
          "a wide city panorama": "landscape", "портрет девушки в полный рост": "portrait"}
ALL_LETTERING = {"убери надпись", "убери все надписи", "remove the lettering"}
AS_FILE = {"пришли эту картинку файлом", "пришли файлом", "send it as a file"}
NO_LETTERING = {'draw a "cat" on a sofa'}

_yes0, _choice0 = intent.YES_STUB, intent.CHOICE_STUB


def _yes(q, t):
    if "Does the instruction remove it" in q:
        lettering = q.split("lettering «", 1)[1].split("»", 1)[0] if "lettering «" in q else ""
        return t in ALL_LETTERING or REMOVES.get((t, lettering), False)
    if "picture on EVERY slide" in q:
        return "каждом слайде" in t or "every slide" in t
    if "as a FILE" in q:
        return t in AS_FILE
    if "written words" in q:
        return t not in NO_LETTERING and "conversation" not in t
    if "OUTERMOST" in q:
        return "крайн" in t or "outermost" in t
    if "repeat, say, rewrite" in q:
        return False
    return _yes0(q, t) if _yes0 else False


def _choice(q, t):
    if "transform a picture" in q:
        return TRANSFORM.get(t, "rotate_right")
    if "put text on a picture" in q:
        return PLACE.get(t, "top")
    if "Which frame fits it?" in q:
        return ORIENT.get(t, "unknown")
    if "What is it about?" in q:
        return KIND.get(t, "code" if "[code]" in t else "image" if "[image]" in t else "chat")
    return _choice0(q, t) if _choice0 else None


intent.YES_STUB, intent.CHOICE_STUB = _yes, _choice
