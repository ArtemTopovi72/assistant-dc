"""Live 2026-10-01: «🔤 Без надписей» was cut to «Без на…» — a row of wide
labels must be split before it reaches Telegram (tg_transport._logged_post)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"))
import tg_transport as T

row = [{"text": t} for t in ("👗 Одежда", "🧽 Убрать", "🔤 Без надписей")]
out = T.fit_rows([row])
assert len(out) > 1, out
assert [b for r in out for b in r] == row, "no button lost or reordered"
short = [[{"text": "Да"}, {"text": "Нет"}]]
assert T.fit_rows(short) == short, "short rows stay"
print("ok")
