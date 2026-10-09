"""10-10 screen review: a model reply's «> quote», «---», «~~x~~» and a nested list item reached
Telegram literally («&gt;», dashes, tildes) or flattened; a fenced block lost its language."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_markup as M

h = M._md_to_html("> цитата **жирно**\n> вторая строка\n\nтекст")
assert h.startswith("<blockquote>цитата <b>жирно</b>\nвторая строка</blockquote>") and "&gt;" not in h, h
assert M._md_to_html("a\n\n---\n\nb") == "a\n\n──────────\n\nb"
assert M._md_to_html("~~нет~~ да") == "<s>нет</s> да"
assert M._md_to_html("- a\n  - b") == "• a\n    ◦ b"
h = M._md_to_html("```python\nx = 1  # **not bold**\n```")
assert '<pre><code class="language-python">x = 1  # **not bold**\n</code></pre>' in h, h
# prose comparisons are not quotes, a «-» list is not a rule
assert "&gt;" in M._md_to_html("5 > 3") and M._md_to_html("- x").startswith("• x")
# a long quote split across messages closes and reopens its tag
parts = M._split_html(M._md_to_html("> " + "слово " * 1200))
assert len(parts) > 1 and all(p.count("<blockquote>") == p.count("</blockquote>") for p in parts), parts
print("PASS md extras")
