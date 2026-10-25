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
note = R.prompt_note([("a VAZ-2107 Lada sedan", "p.jpg")])
check("the note points at image 1", "image 1" in note and "VAZ-2107" in note, note)
check("only the object comes from the photo", "angle, colour, place" in note, note)
check("no things -> no note", R.prompt_note([]) == "")

# The photos go into the Klein graph as reference latents and the prompt says what they are.
seen = {}
def submit(ctx, wf, **kw):
    seen["wf"] = wf; return save("out.png")
cap = {"high_level_description": "a car", "style_description": "photo",
       "compositional_deconstruction": {"background": "street", "elements": []}}
draw = NS(draw_refs=[("a VAZ-2107 Lada sedan", save("ref.jpg"))],
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
