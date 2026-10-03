"""A keyboard button's label must never be 'fixed' as a wrong layout (🎵 Songs -> 🎵 Ыщтпы)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_strings, tg_dispatch, keyboard_layout
labels = [v for f in tg_strings._BTN.values() for v in f.values()]
assert all(tg_dispatch._is_button(l) for l in labels)
assert not tg_dispatch._is_button("ghbdtn")
bad = [l for l in labels if keyboard_layout.fix(l) != l]
assert not bad, bad                          # the detector itself leaves every label alone too
print("PASS layout skips buttons; labels the bare detector mangles:", bad)
