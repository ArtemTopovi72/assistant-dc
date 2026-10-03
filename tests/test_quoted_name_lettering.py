"""A Cyrillic name the user quoted is lettered verbatim (live: «Колосок» drawn as 'Kolosok')."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tool_image_handlers as H

st = {"user_input_original": "нарисуй три логотипа для пекарни «Колосок»"}
assert "«Колосок»" in H._keep_quoted_names("logos for a bakery named 'Kolosok'", st)
assert "Kolosok" not in H._keep_quoted_names("logos for a bakery named 'Kolosok'", st)
assert H._keep_quoted_names("logos, wheat.", st).endswith("«Колосок» in Cyrillic letters.")
assert H._keep_quoted_names("a cat", {"user_input_original": "нарисуй кота"}) == "a cat"
print("PASS quoted name lettering")
