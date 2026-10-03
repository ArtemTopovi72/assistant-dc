"""Can wording beat FireRed's '2025' prior? 3 phrasings x 2 seeds, OCR-scored."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import night_firered as F
SRC = "runtime/_working_input_1785745490393.png"
P = {
 "A": "Add a large bold title in the sky in white poster lettering. The title reads УРОЖАЙ 2026 -- "
      "the year 2026, whose last digit is 6 (six), not 5.",
 "B": 'Add a large bold title "УРОЖАЙ 2026" in the sky in white poster lettering. The number is written '
      'with four digits in this order: 2, 0, 2, 6.',
 "C": 'Write the text "УРОЖАЙ 2026" in large white bold poster letters across the sky. Copy the text '
      'character by character exactly as given; do not change any letter or digit.',
}
base = json.load(open(os.path.join(F.ROOT, "workflows", "image", "workflow_firered_edit.json"), encoding="utf-8"))
d = os.path.join(F.OUT, "title_prompts"); os.makedirs(d, exist_ok=True)
for k, p in P.items():
    wf = F.variant(base, "bf16")
    wf["143"]["inputs"]["image"] = F.upload(os.path.join(F.ROOT, SRC))
    wf["187"]["inputs"]["prompt"] = p
    for r in range(2):
        wf["189"]["inputs"]["seed"] = 43 + r
        img, dt = F.run(wf)
        open(os.path.join(d, f"{k}_r{r}.png"), "wb").write(img)
        print(k, r, round(dt, 1), flush=True)
