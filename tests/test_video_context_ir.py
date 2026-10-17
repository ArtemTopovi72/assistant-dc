"""H3 Context-IR prompt rewrite: accepted only with the required fields, fields
separated by blank lines, on by default in the app and off inside suites."""
from types import SimpleNamespace as NS
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"

import video as V

ok = []


def check(c, m):
    ok.append(bool(c)); print(("ok   " if c else "FAIL ") + m)


os.environ.pop("VIDEO_CONTEXT_IR", None)
check(V._context_ir_on() is False, "off inside suites by default")
os.environ["VIDEO_CONTEXT_IR"] = "1"
check(V._context_ir_on() is True, "explicit 1 turns it on")
os.environ.pop("F5_TEST_RUN")
os.environ.pop("VIDEO_CONTEXT_IR")
check(V._context_ir_on() is True, "on by default in the app")
os.environ["F5_TEST_RUN"] = "1"

good = ("integrated_multimodal_description: [Shot 1] Live-action, a man...\n"
        "overall_soundscape: birds.\nnon_diegetic_music: N/A")
_say = NS(text_model=lambda *a, **k: good)
out = V.to_context_ir(_say, "a man", mode="t2va", seconds=5.17)
check(out.startswith("integrated_multimodal_description:"), "a well-formed rewrite is used")
check("\n\noverall_soundscape:" in out and "\n\nnon_diegetic_music:" in out, "fields separated by a blank line")

_say.text_model = lambda *a, **k: "Sure! Here is a great video idea about a man."
check(V.to_context_ir(_say, "a man", mode="t2va", seconds=5) == "a man", "chatter is rejected, plain prompt kept")
_say.text_model = lambda *a, **k: None
check(V.to_context_ir(_say, "a man", mode="t2va", seconds=5) == "a man", "no answer -> plain prompt kept")

ref_ok = ("subject_definitions:\n<Subject 1> is ...\nsummary:\nx\nretention_analysis:\ny\n"
          "detailed_description:\n[Shot 1] ...\noverall_soundscape: z\nnon_diegetic_music: N/A")
seen = {}
look = NS(vision_model=lambda ctx, image_paths=None, user_text="", system_prompt="", **k:
         seen.update(img=[p for _, p in image_paths or []], sys=system_prompt[:60], user=user_text) or ref_ok)
out = V.to_context_ir(look, "<Picture 1> walks", mode="ref2va", seconds=5, images=["p.png", "guy.png"])
check(out.startswith("subject_definitions:"), "Ref2VA six-section rewrite accepted")
# Shown only the first, the writer guessed the newcomer of <Picture 2> («a dark glove» for
# a light-blue one, only his arm in the shot; 2026-10-12).
check(seen.get("img") == ["p.png", "guy.png"], "the rewriter SEES every reference picture")
check("Full-Reference" in seen.get("sys", ""), "Ref2VA uses the full-reference guide")
# A plan part naming only the newcomer («a man lunges out of the cabinet») left the
# grandma of the first frame out, and H3 rendered the room without her (2026-10-12).
check("every person and animal visible in <Picture 1> is in the shot" in seen.get("user", ""),
      "whoever is in the first frame stays in the shot")
# «He turns and shouts» with no words came back as <d>[English] Look at that!</d> (2026-10-15).
check("never a line you make up" in seen.get("user", ""), "the writer adds no spoken line of its own")
# Part 2 of the beach clip opened on «a wide shot frames two tiny figures» (a jump cut) and
# added «a third figure runs through the mist» -- the girl of part 1, running past again.
check("never a cut and never a second [Shot]" in seen.get("user", ""), "a continued shot is one shot, moved in real time")
check("never enters, runs in or appears" in seen.get("user", ""), "nobody already in the scene comes in again")
# A traveler seen from behind, «whispers in a low tone», spoke in a woman's whisper in part 1
# and as a man in part 2 (2026-10-15).
check("names the speaker's sex in its voice" in seen.get("user", "")
      and "same voice in every part" in seen.get("user", ""), "a voice is a man's or a woman's, kept across parts")
# Told in words, the writer still opened part 2 with «a medium shot frames a traveler»: a jump.
reframed = ("integrated_multimodal_description: [Shot 1] Live-action, a medium shot frames a traveler. "
            "The camera holds a static shot as he drinks.\noverall_soundscape: wind.\nnon_diegetic_music: N/A")
fixed = reframed.replace("a medium shot frames a traveler", "the traveler, framed as the clip ends,")
answers = iter([reframed, fixed])
asked = []
cont = V.to_context_ir(NS(text_model=lambda *a, **k: next(answers),
                          yes_no=lambda q, text, **k: asked.append(text) or "a medium shot frames" in text),
                       V.CONTINUE_CTX_PREFIX + "He drinks.", mode="t2va", seconds=5)
check(asked and cont.startswith("integrated_multimodal_description: [Shot 1] Live-action, the traveler, framed"),
      "a continued part the model reads as a new framing is rewritten once")
asked.clear()
first = V.to_context_ir(NS(text_model=lambda *a, **k: reframed,
                           yes_no=lambda q, text, **k: asked.append(text) or True),
                        "He drinks.", mode="t2va", seconds=5)
check("a medium shot frames" in first and not asked, "a first part keeps its framing, nothing is asked")
# A part with no words came back with «A distant man's voice calls: <d>[English] Hello!</d>».
hello = ("integrated_multimodal_description: [Shot 1] Live-action, the hiker stands. He says "
         "<d>[Russian] Наконе́ц-то я здесь.</d> A distant man's voice calls: <d>[English] Hello!</d>\n"
         "overall_soundscape: wind.\nnon_diegetic_music: N/A")
out = V.to_context_ir(NS(text_model=lambda *a, **k: hello), 'The hiker says "Наконец-то я здесь".',
                      mode="t2va", seconds=5)
check("Hello" not in out and "Наконе́ц-то я здесь" in out and "a wordless call" in out,
      "a spoken line the request never quoted is made wordless; the quoted one stays")
# «говорит несколько раз «Куришь?»» came back as one <d> line of three and was cut as made up
thrice = ("integrated_multimodal_description: [Shot 1] Live-action, a man says "
          "<d>[Russian] Ку́ришь? Ку́ришь? Ку́ришь?</d>\noverall_soundscape: room.\nnon_diegetic_music: N/A")
out = V.to_context_ir(NS(text_model=lambda *a, **k: thrice), 'He says several times: "Куришь?"',
                      mode="t2va", seconds=5)
check("Ку́ришь? Ку́ришь? Ку́ришь?" in out and "wordless" not in out, "the request's line said several times stays")
# «кряхтит» written as <d>Хррр-ммм!</d> was cut and the clip came back silent
grunt = ("integrated_multimodal_description: [Shot 1] Live-action, a man grunts "
         "<d>[Russian] Хррр-ммм!</d>\noverall_soundscape: room.\nnon_diegetic_music: N/A")
out = V.to_context_ir(NS(text_model=lambda *a, **k: grunt,
                         yes_no=lambda q, text, **k: "name this kind of sound" in q and "grunts" in text),
                      "A man grunts heavily", mode="t2va", seconds=5)
check("Хррр-ммм" in out, "a grunt the request asks for keeps its voice")
yawn = ("integrated_multimodal_description: [Shot 1] Live-action, he yawns <d>[English] ...</d> and stops."
        "\noverall_soundscape: room.\nnon_diegetic_music: N/A")
out = V.to_context_ir(NS(text_model=lambda *a, **k: yawn), "He yawns", mode="t2va", seconds=5)
check("<d>" not in out and "he yawns and stops" in out, "a voiced line with nothing to say is dropped")
said = ("integrated_multimodal_description: [Shot 1] Live-action, he reads. His eyes widen, and the man (S1) "
        "whispers: <d>[English] ...</d> while frozen in shock. He smiles.\noverall_soundscape: room.\nnon_diegetic_music: N/A")
out = V.to_context_ir(NS(text_model=lambda *a, **k: said), "He reads and smiles", mode="t2va", seconds=5)
check("whispers" not in out and "he reads." in out and "He smiles." in out,
      "...and «whispers: <empty>» goes with its sentence, not left as «whispers: while»")
# the accentor also restores ё its own way: «тише, все спят» was spoken «всё спят»
check(V._marks_only("тише, все спят", "ти́ше всё спят") == "ти́ше, все спят",
      "stress marks only: letters and punctuation stay as the user wrote them")
check(V._marks_only("ну ты даёшь", "ну ты дае́шь") == "ну ты даёшь",
      "an ё the user wrote stays")
check(V.to_context_ir(NS(vision_model=lambda *a, **k: good), "x", mode="ref2va", seconds=5, images=["p.png"]) == "x",
      "a base-mode answer is not accepted for Ref2VA")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
