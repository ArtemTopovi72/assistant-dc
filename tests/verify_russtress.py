import os; os.environ["TF_USE_LEGACY_KERAS"] = "1"
from russtress import Accent
rs = Accent()

APOS = frozenset(["'", "’"])  # apostrophe chars russtress uses

def apos_to_plus(text):
    out = []
    i = 0
    while i < len(text):
        if i + 1 < len(text) and text[i + 1] in APOS:
            out.append("+")
            out.append(text[i])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)

tests = [
    "электроудочники незаконно ловят рыбу.",
    "Государственная дума приняла закон.",
    "Самовоспламенение горючих материалов.",
    "Высокопроизводительный суперкомпьютер.",
]

for t in tests:
    raw = rs.put_stress(t)
    converted = apos_to_plus(raw)
    print(f"IN:  {t}")
    print(f"RAW: {raw}")
    print(f"OUT: {converted}")
    print()
