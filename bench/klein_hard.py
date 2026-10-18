"""Hard layout scenes: Ideogram 4 vs Klein driven by the SAME Ideogram caption (auto-converted).

Each scene is an imaging.ideogram_layout caption (8-10 boxed elements, dense Russian text, overlapping
boxes). Klein gets the same caption turned into its native input: a paragraph citing <ids> + a JSON table
{id, bbox, desc} on the 0-1000 grid (ig_to_klein). Same seeds, 1024x1024.

    venv/Scripts/python bench/klein_hard.py --variants ig_app,ig_full,kl_int8
Output: runtime/klein_hard/<variant>/s<N>_<name>.png + timings.json
"""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, ROOT)
import night_firered  # noqa: E402
import klein_vs_ideogram as K  # noqa: E402
from imaging.ideogram_layout import build_caption, element  # noqa: E402

OUT = os.path.join(ROOT, "runtime", "klein_hard")


def E(desc, box, text=""):
    return element(desc, box, text=text) if text else element(desc, box)


SCENES = [
    ("market", build_caption("an open-air farmers market in a town square on a sunny morning", [
        E("large hand-painted wooden arch sign over the market entrance, red letters on cream", [30, 300, 130, 700], "РЫНОК"),
        E("chalkboard price tag on the left stall, white chalk handwriting", [560, 60, 640, 300], "ЯБЛОКИ 120 ₽"),
        E("small wooden plate on the right stall, burnt-in lettering", [560, 700, 630, 860], "МЁД"),
        E("woman vendor in a red apron standing behind the left stall, smiling", [300, 60, 640, 330]),
        E("old man in a grey flat cap buying apples, seen from the front, holding a string bag", [350, 380, 900, 600]),
        E("boy in a bright yellow raincoat pointing at the honey jars", [450, 700, 950, 900]),
        E("wooden crate full of red apples", [640, 60, 820, 330]),
        E("old green bicycle with a wicker basket leaning on the right", [520, 860, 990, 990]),
        E("small brown dog sitting on the cobblestones", [820, 440, 980, 600]),
        E("red helium balloon floating above the crowd", [140, 600, 300, 700])],
        high_level="A busy farmers market with three people, a dog, a bicycle and hand-lettered signs.",
        aesthetics="warm documentary", lighting="bright morning sun", photo="35mm street photo")),
    ("bookshop", build_caption("a cozy old bookshop interior with warm lamps", [
        E("brass plate above the left bookshelf with engraved capital letters", [60, 40, 140, 330], "ФАНТАСТИКА"),
        E("brass plate above the middle bookshelf with engraved capital letters", [60, 380, 140, 620], "ИСТОРИЯ"),
        E("brass plate above the right bookshelf with engraved capital letters", [60, 680, 140, 960], "ДЕТЯМ"),
        E("tall wooden bookshelf packed with colourful spines", [150, 30, 700, 340]),
        E("tall wooden bookshelf packed with old leather books", [150, 370, 700, 640]),
        E("tall wooden bookshelf packed with picture books", [150, 670, 700, 970]),
        E("middle-aged bookseller with round glasses and a green cardigan behind the counter, reading a book", [520, 420, 900, 640]),
        E("grey cat sleeping curled up on the wooden counter", [740, 660, 860, 860]),
        E("round wall clock with a white dial", [180, 470, 300, 560]),
        E("stack of hardcover books on the floor", [820, 60, 960, 300])],
        high_level="A bookshop with three labelled shelves, a bookseller and a sleeping cat.",
        aesthetics="cozy", lighting="warm lamp light", photo="35mm interior photo")),
    ("station", build_caption("a railway platform in the evening rain, wet asphalt reflecting lights", [
        E("large white station sign with bold capital letters on a dark blue board", [40, 350, 150, 650], "ВОКЗАЛ"),
        E("electronic departure board with amber letters on black, two lines", [180, 720, 320, 960],
          "Поезд № 7 — Москва\nОтправление 18:45"),
        E("round station clock with black hands", [60, 120, 180, 260]),
        E("young woman in a beige coat holding a red umbrella", [380, 60, 900, 260]),
        E("man in a dark overcoat pulling two brown suitcases", [420, 380, 900, 560]),
        E("child in a yellow rain jacket with a small blue backpack", [560, 600, 900, 720]),
        E("blue passenger train with lit windows standing at the platform", [300, 740, 850, 1000]),
        E("grey pigeon walking on the wet platform", [880, 300, 960, 380])],
        high_level="An evening platform with a station sign, a departure board, three travellers and a train.",
        aesthetics="moody cinematic", lighting="rainy dusk with sodium lamps", photo="50mm photo")),
    ("menu", build_caption("a black chalkboard menu hanging on a red brick wall", [
        E("large chalk title, white", [30, 380, 130, 620], "МЕНЮ"),
        E("chalk line of the menu", [170, 100, 240, 900], "Борщ — 280 ₽"),
        E("chalk line of the menu", [260, 100, 330, 900], "Пельмени — 350 ₽"),
        E("chalk line of the menu", [350, 100, 420, 900], "Блины с мёдом — 220 ₽"),
        E("chalk line of the menu", [440, 100, 510, 900], "Чай с лимоном — 120 ₽"),
        E("chalk line of the menu", [530, 100, 600, 900], "Кофе — 150 ₽"),
        E("chalk drawing of a steaming teapot", [630, 380, 790, 620]),
        E("chalk greeting, slightly slanted", [820, 250, 930, 750], "Приятного аппетита!")],
        high_level="A cafe chalkboard menu with five priced dishes, a title and a greeting.",
        aesthetics="rustic", lighting="soft side light", photo="50mm photo")),
    ("toys", build_caption("a wooden toy-shop shelf with two rows of toys", [
        E("painted wooden header sign", [30, 300, 140, 700], "ИГРУШКИ"),
        E("brown teddy bear with a red bow", [180, 40, 480, 230]),
        E("small red toy car", [260, 260, 480, 450]),
        E("silver tin robot with round eyes", [170, 480, 480, 660]),
        E("ballerina doll in a pink tutu", [170, 700, 480, 860]),
        E("wooden rocking horse", [540, 40, 900, 260]),
        E("yellow rubber duck", [660, 300, 860, 440]),
        E("red toy drum with white sticks", [580, 480, 900, 690]),
        E("colourful paper kite hanging on the right", [540, 740, 940, 960])],
        high_level="A toy shop shelf with eight toys placed in two rows under a sign.",
        aesthetics="playful", lighting="soft daylight", photo="50mm product photo")),
]


def pos_phrase(b):
    y = (b[0] + b[2]) / 2000
    x = (b[1] + b[3]) / 2000
    v = "upper" if y < .34 else "lower" if y > .66 else "middle"
    h = "left" if x < .34 else "right" if x > .66 else "centre"
    return f"{v} {h}"


def ig_to_klein(cap):
    """Ideogram caption -> Klein input: paragraph citing <ids> + JSON element table."""
    comp = cap["compositional_deconstruction"]
    st = cap.get("style_description", {})
    ids, table, cites = [], [], []
    for n, el in enumerate(comp["elements"], 1):
        if el["type"] == "text":
            i = f"Ru_Text_{sum(1 for e in comp['elements'][:n] if e['type'] == 'text')}"
        else:
            w = re.sub(r"[^a-z]", "", (el["desc"].split(" of ")[0].split()[-1] if " of " in el["desc"] else el["desc"].split()[0]).lower())
            i = f"{w or 'object'}_{n}"
        desc = el["desc"] + (f'. Text: "{el["text"]}"' if el.get("text") else "")
        table.append({"id": i, "bbox": el["bbox"], "desc": desc})
        cites.append(f"{pos_phrase(el['bbox'])}: {el['desc']} <{i}>")
    medium = "Full-color photograph" if st.get("medium") == "photograph" else "Illustration"
    para = (f"{medium}, {cap['high_level_description']} Setting: {comp['background']}. "
            f"Lighting: {st.get('lighting', '')}; {st.get('photo', '')}. " + ". ".join(c[0].upper() + c[1:] for c in cites) + ".")
    return para + " " + json.dumps(table, ensure_ascii=False, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="ig_app,ig_full,kl_int8")
    ap.add_argument("--comfy-klein", default="http://127.0.0.1:8010")
    a = ap.parse_args()
    base = json.load(open(os.path.join(ROOT, "workflows", "image", "workflow_ideogram4.json"), encoding="utf-8"))
    for v in a.variants.split(","):
        d = os.path.join(OUT, v)
        os.makedirs(d, exist_ok=True)
        night_firered.COMFY = a.comfy_klein if v.startswith("kl") else "http://127.0.0.1:8000"
        res = []
        for i, (name, cap) in enumerate(SCENES):
            seed = 2000 + i
            try:
                if v.startswith("ig"):
                    wf = K.ig_wf(base, cap, seed, v)
                else:
                    wf = K.klein_wf(ig_to_klein(cap), seed, *K.KLEIN["kl_dist_int8"])
                img, dt = night_firered.run(wf)
                p = os.path.join(d, f"s{i}_{name}.png")
                open(p, "wb").write(img)
                res.append({"scene": name, "seconds": round(dt, 1), "file": p})
                print(f"{v} {name} {dt:.1f}s", flush=True)
            except Exception as exc:
                res.append({"scene": name, "error": str(exc)[:600]})
                print(f"{v} {name} ERROR {exc}", flush=True)
        json.dump(res, open(os.path.join(d, "timings.json"), "w"), indent=1, ensure_ascii=False)
    if os.environ.get("PRINT_PROMPT"):
        print(ig_to_klein(SCENES[0][1]))


if __name__ == "__main__":
    main()
