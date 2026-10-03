import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The model's reads are stubbed here; the phrases run live in bench/intent_rest_live.py.
import intent
intent.YES_STUB = lambda q, t: "NO singing" in q and any(w in t.lower() for w in ("без вокала", "без слов", "instrumental", "инструментал"))
import llm, music

for genre, secs, want_g, want_s in (("lifi", 120, "lofi", 120), ("hip-hop", 15, "hiphop", 20),
                                    ("R&B", 400, "rnb", 180), ("polka", 60, None, 60)):
    llm.call_llm_simple = lambda *a, **k: json.dumps({"genre": genre, "bpm": 70, "vocal": "female",
                                                      "seconds": secs, "why": "x"})
    got = music.choose_auto_params(None, "t", "ru", genre=True, tempo=True, vocal=True, duration=True)
    assert got.get("genre") == want_g and got["seconds"] == want_s, (genre, got)
print("PASS auto picks: near-miss genres and out-of-range lengths are kept, not dropped")
got = music.choose_auto_params(None, "короткий джингл, бодрый, без вокала", "ru", vocal=True)
assert got.get("vocal") == "instrumental", got
assert music.prefs_from(vocal=got["vocal"]) == {"instrumental": True}
print("PASS 'без вокала' is picked as instrumental")
