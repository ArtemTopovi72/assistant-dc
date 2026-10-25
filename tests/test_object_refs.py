"""A named real thing is drawn from a checked web photo of it.

Klein knows the kind of thing, not the model: «ВАЗ 2107» came out as a Fiat, the «Урал»
sidecar as a fuel tank; with a photo of it as a reference latent the same seed drew a true
2107 (debug chat + bench/object_ref_probe.py, 2026-10-25).
"""
import os, sys, tempfile, types
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from PIL import Image
import ideogram as I
import klein
import object_refs as R

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

tmp = tempfile.mkdtemp(prefix="objrefs_")
NS = types.SimpleNamespace

# Suites draw offline: no model read, no web search unless a suite asks for it.
check("no lookup inside suites", R.find(NS(), "ВАЗ 2107 на улице") == [])

# A candidate that is not the thing is skipped; the one that is gets cut to its box --
# the street and the stray stick around the «Урал» were copied along with it.
def save(name, size=(1000, 500)):
    p = os.path.join(tmp, name); Image.new("RGB", size, (90, 120, 90)).save(p); return p
photos = iter([save("ad.jpg"), save("ural.jpg")])
looked = []
def look(ctx, image_paths=None, user_text="", **k):
    looked.append(user_text)
    return '{"shows": false}' if len(looked) == 1 else '{"shows": true, "box": [200, 100, 700, 900]}'
def download(ctx, url, dest):
    Image.open(next(photos)).save(dest); return True
ctx = NS(vision_model=look, deps=NS(image_search_urls=lambda ctx, q, prefer="": ["u1", "u2", "u3"],
                                    download_image=download))
got = R.photo_of(ctx, "a Ural motorcycle with a sidecar", "Урал с коляской")
check("a photo that does not show the thing is skipped", len(looked) == 2, looked)
check("the thing is named in the check", "Ural motorcycle" in looked[0])
w, h = Image.open(got).size if got else (0, 0)
check("the photo is cut to the thing, with a margin", 500 < w < 600 and 400 < h < 500, (w, h))

# Nothing found, or the search refused (weapons go nowhere): no photo, the words draw it.
check("no candidates -> no photo",
      R.photo_of(NS(vision_model=look, deps=NS(image_search_urls=lambda *a, **k: [])), "x", "y") == "")

# The note names what each reference is for, and that only the object is taken from it:
# the angle, colour and place the request asks for are the picture's own.
note = R.prompt_note([("a VAZ-2107 Lada sedan", "p.jpg", "object")])
check("the note points at image 1", "image 1" in note and "VAZ-2107" in note, note)
check("only the object comes from the photo", "angle, colour, place" in note, note)
check("no things -> no note", R.prompt_note([]) == "")
# A person gives only the face: Lenin in Doom armour keeps his head, not his suit.
note = R.prompt_note([("a VAZ-2107 Lada sedan", "v.jpg", "object"), ("the Doom Slayer's armour", "d.jpg", "character")])
check("the second reference is image 2", "image 2" in note, note)
# A character's colours are part of its look; a car's paint is the picture's own.
check("a character keeps its colours", "angle, pose, place" in note.split("image 2")[1], note)
check("a car's colour is the picture's", "angle, colour, place" in note.split("image 2")[0], note)

# A face is read off its photo, not off web text: the web words for Lenin said «often wears
# glasses» and he was drawn in red sunglasses (debug chat 2026-10-26).
import tool_image_handlers as H
seen_desc = {}
def fake_render(ctx=None, description="", **k):
    seen_desc["d"] = description
    return {"path": None, "status": "fail"}
def run(refs):
    seen_desc.clear()
    c = NS(deps=NS(object_refs=lambda ctx, d: refs, generate_image_with_refinement=fake_render,
                   _render_budget_exhausted=lambda *a: None,
                   face_words=lambda ctx, who, p: f"{who} must be recognisable: bald, a short pointed goatee",
                   person_look=lambda text: ("Vladimir Lenin", "bald, small reddish beard, often wears glasses")),
           set_stage=lambda *a: None)
    try:
        H._handle_generate_image(c, {"user_input": "нарисуй Ленина"}, {"description": "Lenin in hell"})
    except Exception as exc:
        seen_desc.setdefault("err", repr(exc))
    return seen_desc.get("d", seen_desc.get("err", ""))
check("no face photo -> the web words describe him", "glasses" in run([]), run([]))
d = run([("Vladimir Lenin", save("l.jpg"), "person")])
check("a face photo -> his look read off it, not the web", "pointed goatee" in d and "glasses" not in d, d)

# The face never goes into the scene render: as a latent it set the frame (Lenin staring into the
# camera, the fight gone). The scene is drawn free, then the face is edited in.
seen = {}
swapped = []
def submit(ctx, wf, **kw):
    seen["wf"] = wf; return save("out.png")
cap = {"high_level_description": "a man", "style_description": "photo",
       "compositional_deconstruction": {"background": "hell", "elements": []}}
lenin = save("lenin.jpg")
out = I.generate(NS(draw_refs=[("Vladimir Lenin", lenin, "person")],
                    deps=NS(_submit_and_poll=submit, _upload_image_to_comfy=lambda p, url: "up.jpg",
                            swap_faces=lambda ctx, p, faces: swapped.append(faces) or save("swapped.png"))),
                 "Ленин", caption=cap, seed=7)
check("the face is not a reference latent of the scene", klein.N_REF not in seen.get("wf", {}))
check("the face is edited into the drawn scene", swapped == [[("Vladimir Lenin", lenin)]], swapped)
check("the edited picture goes out", out.endswith("swapped.png"), out)

# Too small a face is left as drawn: no edit makes a rider's few-pixel face more anyone.
edits = []
small = R.swap_faces(NS(deps=NS(edit_image=lambda *a, **k: edits.append(a) or "e.png")), save("tiny.png"),
                     [("Vladimir Lenin", lenin)])
check("no face big enough -> no edit", not edits and small.endswith("tiny.png"), (edits, small))

# The photos go into the Klein graph as reference latents and the prompt says what they are.
seen = {}
def submit(ctx, wf, **kw):
    seen["wf"] = wf; return save("out.png")
cap = {"high_level_description": "a car", "style_description": "photo",
       "compositional_deconstruction": {"background": "street", "elements": []}}
draw = NS(draw_refs=[("a VAZ-2107 Lada sedan", save("ref.jpg"), "object")],
          deps=NS(_submit_and_poll=submit, _upload_image_to_comfy=lambda p, url: "ref_up.jpg"))
I.generate(draw, "ВАЗ 2107", caption=cap, seed=7)
wf = seen.get("wf", {})
ref = wf.get(klein.N_REF, {}).get("inputs", {})
check("the photo is a reference latent", ref.get("images.image0") == ["ref_load_0", 0]
      and wf.get("ref_load_0", {}).get("inputs", {}).get("image") == "ref_up.jpg", ref)
check("the sampler is conditioned through it", wf.get(klein.N_SAMPLER, {}).get("inputs", {}).get("positive") == [klein.N_REF, 0])
check("the prompt says what the reference is", "image 1" in wf.get(klein.N_POS, {}).get("inputs", {}).get("text", ""))
seen.clear()
I.generate(NS(deps=NS(_submit_and_poll=submit)), "a car", caption=cap, seed=7)
check("no things -> the plain graph", klein.N_REF not in seen.get("wf", {}))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
