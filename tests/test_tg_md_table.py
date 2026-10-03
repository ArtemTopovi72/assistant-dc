"""Live 2026-09-28: a markdown table reached Telegram as raw '| --- |' pipes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_markup as M

h = M._md_to_html("Вот:\n\n| A | Bb |\n| --- | --- |\n| 1 | **x** |\n| 22 | y |\nконец")
assert "<pre>" in h and "---" not in h and "|" not in h and "**" not in h, h
assert "22  y" in h, h
# a table already inside a code fence is left as the author wrote it
h2 = M._md_to_html("```\n| a | b |\n| - | - |\n```")
assert "| a | b |" in h2, h2
# a lone pipe in prose is not a table
assert M._md_to_html("a | b") == "a | b"
# backticks inside a fenced block stay literal; outside they are code
h3 = M._md_to_html("```\nx `y`\n```\nand `z`")
assert "`y`" in h3 and "<code>z</code>" in h3, h3
print("PASS md table")

