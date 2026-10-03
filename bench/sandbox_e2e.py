"""End to end: a real modpack, the real agent loop, the real tools.

Everything below the model is REAL here — the sandbox, the archive, the file
edits, the zip that comes out the other side. Only the model is being measured,
because everything else already has a unit suite and the open question is
different: given tools that genuinely work, does a local 26B actually USE them,
or does it go back to asking the user to paste source code at it?

That question cannot be answered with stubs. A stub records that unpack_archive
was called; it cannot tell us whether the model then found the right file among
two hundred, whether it copied the JSON exactly enough for an exact-match edit
to apply, or whether it remembered to pack the result at the end. Those are the
three places this actually breaks.

Scored on the OUTCOME, not the transcript: did the user end up with an archive
containing the change they asked for? A run that calls every tool in the right
order and delivers nothing has failed.

Usage:
    venv/Scripts/python.exe bench/sandbox_e2e.py [--reps 3] [--case tag_edit]
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tc_run import setup                    # noqa: E402  (shared bench boot)
import code_sandbox as CS                   # noqa: E402
from code_sandbox import AGENT_DIR          # noqa: E402
import sandbox_access as A                  # noqa: E402


class _User:
    """A granted user. `code` rather than `host`: the bench must measure what a
    normally-configured account can do, not what the owner can."""
    prefs = {"sandbox": A.CODE}


# ── fixtures ────────────────────────────────────────────────────────────────

def _thief_jar(box: CS.Sandbox, name="thief-1.21.1.jar") -> str:
    """A mod whose behaviour is data-driven, like the real one.

    Padded with plausible neighbours on purpose. With one file in the archive
    the model cannot help but find it; the actual difficulty is picking the
    right one of several files that all look relevant, which is where a
    real modpack puts the work.
    """
    with zipfile.ZipFile(box.root / name, "w") as zf:
        base = "data/thief/tags/block"
        zf.writestr(f"{base}/break_protected/light.json",
                    json.dumps({"values": ["#c:glass_panes"]}, indent=2))
        zf.writestr(f"{base}/break_protected/medium.json",
                    json.dumps({"values": ["#c:chests", "#c:barrels",
                                           "#c:villager_job_sites",
                                           "#minecraft:beds"]}, indent=2))
        zf.writestr(f"{base}/break_protected/heavy.json",
                    json.dumps({"values": ["#c:shulker_boxes"]}, indent=2))
        zf.writestr(f"{base}/interact_protected/medium.json",
                    json.dumps({"values": ["#c:chests"]}, indent=2))
        zf.writestr(f"{base}/killing_protected/heavy.json",
                    json.dumps({"values": []}, indent=2))
        zf.writestr("pack.mcmeta",
                    json.dumps({"pack": {"description": "Thief",
                                         "pack_format": 48}}, indent=2))
        zf.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")
    return name


def _two_mods(box: CS.Sandbox) -> str:
    """The case from the chat that started this: the answer is in NEITHER jar.

    Thief's tag says which tag is protected (#c:villager_job_sites); the block
    ids that carry that tag live in the OTHER mod. Answering means opening both
    archives and joining them -- which is the step the assistant that prompted
    this work refused to do at all, asking the user to paste files instead.
    """
    _thief_jar(box)
    with zipfile.ZipFile(box.root / "morevillagers-1.21.1.jar", "w") as zf:
        zf.writestr("data/morevillagers/tags/block/villager_job_sites.json",
                    json.dumps({"values": ["morevillagers:fletching_table_2",
                                           "morevillagers:trading_table",
                                           "morevillagers:hunter_table"]}, indent=2))
        # No data/c/... bridge here, and that is the whole point of the
        # fixture. An earlier version shipped one pointing at
        # "#morevillagers:job_sites" -- a tag id that does not exist, since the
        # file above is villager_job_sites.json. So Thief's medium.json, which
        # already lists #c:villager_job_sites, appeared to cover these blocks
        # through a reference that dangles. The task was then unanswerable and
        # the bench blamed the model for it: three runs delivered a correct
        # archive that added "#morevillagers:job_sites", faithfully copying the
        # broken name the fixture had shown them.
        #
        # Without the bridge the chain is genuinely open, so there is a real
        # fix and exactly two reasonable ways to write it -- see the check.
        zf.writestr("pack.mcmeta", json.dumps({"pack": {"pack_format": 48}}))
    return "morevillagers-1.21.1.jar"


def _check_cross_mod(box, state):
    answer = (state.get("final_answer") or "").lower()
    named = [b for b in ("trading_table", "hunter_table", "fletching_table_2")
             if b in answer]
    if len(named) >= 2:
        return True, "joined the two archives: named " + ", ".join(named)
    if named:
        return False, f"found only {named[0]} of three: {answer[:90]}"
    if "прислать" in answer or "пришли" in answer or "вставь" in answer:
        return False, f"asked for files it already has: {answer[:90]}"
    return False, f"never joined the two mods: {answer[:130]}"


def _check_deliver_fix(box, state):
    """The chat case that started this: a RESULT was asked for, not a file.

    Same two jars as cross_mod, but the request names the symptom rather than
    the steps -- no "запакуй", no "пришли файл". Live over Telegram the agent
    answered this correctly IN PROSE, including the ten-line JSON, and told the
    user to create the folders by hand; it only wrote and packed anything after
    a second message saying "сделай сам и отправь готовый файл". So prose has
    to FAIL here however correct it is: the user asked for the villagers to
    work again, and a message they must retype is not that.
    """
    doc = str(state.get("document_path") or "")
    if not doc or not Path(doc).exists():
        return False, ("answered without delivering anything: "
                       + (state.get("final_answer") or "")[:120])
    if state.get("document_status") != "success":
        return False, ("built %s but document_status=%r, which the Telegram "
                       "delivery block does not send"
                       % (Path(doc).name, state.get("document_status")))
    if not zipfile.is_zipfile(doc):
        return False, f"delivered {Path(doc).name}, which is not an archive"
    # Deliberately not pinned to one path: the fix is legitimate either as an
    # edit of Thief's own tag or as a new datapack overriding it, and both were
    # produced by correct runs. What is NOT negotiable is that some JSON in the
    # delivered archive lists the morevillagers workstations.
    want = ("trading_table", "hunter_table", "fletching_table_2")
    # Either way of writing the fix counts, because both are correct Minecraft:
    # naming the three blocks outright, or referencing the tag that already
    # holds them (#morevillagers:villager_job_sites). What does NOT count is a
    # reference to a tag that does not exist -- that reads as a fix and
    # protects nothing, which is exactly the failure the old fixture provoked.
    real_tag = "morevillagers:villager_job_sites"
    with zipfile.ZipFile(doc) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            # It has to be THIEF's protection list. The first version of this
            # check accepted any JSON in the archive that named the blocks, and
            # scored 3/3 on runs that repacked the morevillagers jar UNTOUCHED
            # -- its own villager_job_sites.json already lists them, so the
            # check found what it was looking for in a file nobody had edited.
            # Protection lives in break_protected; anywhere else changes
            # nothing about whether stealing is noticed.
            # The third correct fix, found by a real run: add the blocks to the
            # COMMON tag Thief already protects (#c:villager_job_sites, listed
            # in the fixture's break_protected/medium.json). Tags merge across
            # mods (replace defaults to false), so this is the idiomatic
            # convention-tag fix and was wrongly scored as a failure.
            if name == "data/c/tags/block/villager_job_sites.json":
                try:
                    data = json.loads(zf.read(name))
                except ValueError:
                    continue
                if data.get("replace") is True:
                    continue
                values = " ".join(str(v) for v in (data.get("values") or []))
                named = [b for b in want if b in values]
                if len(named) >= 2:
                    return True, "delivered %s: %s adds %s to the tag Thief protects" % (
                        Path(doc).name, name, ", ".join(named))
                if real_tag in values:
                    return True, "delivered %s: %s adds #%s to the tag Thief protects" % (
                        Path(doc).name, name, real_tag)
                continue
            if "break_protected" not in name:
                continue
            try:
                data = json.loads(zf.read(name))
            except ValueError:
                continue
            values = " ".join(str(v) for v in (data.get("values") or []))
            named = [b for b in want if b in values]
            if len(named) >= 2:
                return True, "delivered %s: %s lists %s" % (
                    Path(doc).name, name, ", ".join(named))
            if real_tag in values:
                return True, "delivered %s: %s references #%s" % (
                    Path(doc).name, name, real_tag)
    return False, (f"{Path(doc).name} protects nothing new — no tag in it "
                   f"lists the morevillagers workstations or references "
                   f"#{real_tag}")


def _photo(box: CS.Sandbox, name="scan.png") -> str:
    """A picture with content nothing can guess from the prompt.

    Drawn rather than shipped: a fixture PNG in the repo would be a binary blob
    nobody can review, and the point is only that the agent must LOOK. A big
    red triangle on white is unambiguous to any vision model and impossible to
    answer correctly by pattern-matching the question.
    """
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(im).polygon([(256, 60), (60, 440), (452, 440)], fill=(220, 30, 30))
    im.save(box.root / name)
    return name


def _check_photo_read(box, state):
    """Did it open the picture and answer from what is IN it?

    Scored on the shape and the colour together: "красный" alone could be luck
    on a question about a picture, and the shape cannot be guessed at all.
    """
    answer = (state.get("final_answer") or "").lower()
    shape = any(w in answer for w in ("треугольник", "triangle"))
    colour = any(w in answer for w in ("красн", "red", "алый"))
    if shape and colour:
        return True, "named the shape and the colour"
    if "пришл" in answer or "отправ" in answer and "картинк" in answer:
        return False, f"asked for a picture it already has: {answer[:90]}"
    if shape or colour:
        return False, f"half an answer: {answer[:110]}"
    return False, f"no evidence it looked: {answer[:120]}"


def _broken_json(box: CS.Sandbox, name="config") -> str:
    box.write_text(f"{name}/settings.json",
                   '{\n  "volume": 0.8,\n  "name": "test",\n}\n')   # trailing comma
    return f"{name}/settings.json"


# ── cases ───────────────────────────────────────────────────────────────────
# `check` receives the sandbox and the final state and answers one question:
# did the USER get what they asked for?

def _check_tag_edit(box, state):
    doc = str(state.get("document_path") or "")
    if not doc or not Path(doc).exists():
        return False, "nothing was delivered"
    # The status the DELIVERY layer requires, not merely a path. Checking the
    # path alone is how this bench reported "delivered Thief_modified.jar" for
    # months while tg_tasks dropped every one of them: pack_archive wrote "ok"
    # and the sender tests for "success".
    if state.get("document_status") != "success":
        return False, ("built %s but document_status=%r, which the Telegram "
                       "delivery block does not send"
                       % (Path(doc).name, state.get("document_status")))
    if not zipfile.is_zipfile(doc):
        return False, f"delivered {Path(doc).name}, which is not an archive"
    with zipfile.ZipFile(doc) as zf:
        names = zf.namelist()
        target = [n for n in names if n.endswith("break_protected/medium.json")]
        if not target:
            return False, "the archive lost the tag file"
        try:
            data = json.loads(zf.read(target[0]))
        except ValueError as exc:
            return False, f"the tag file is no longer valid JSON: {exc}"
    values = " ".join(str(v) for v in data.get("values", []))
    if "morevillagers" not in values:
        return False, f"the block id never made it in: {values[:120]}"
    if "villager_job_sites" not in values:
        return False, "it added the new id but dropped the existing ones"
    return True, f"delivered {Path(doc).name} with the tag edited"


def _check_read_and_answer(box, state):
    """Did it open the archive and answer from what is inside?

    The first version of this demanded the literal string "villager_job_sites"
    and failed a run whose answer was "отвечает тег medium в папке
    break_protected" -- which is correct, and could only have been written by
    something that had opened the jar. The scorer was wrong, not the agent.

    So: evidence of having READ it. Either a value out of the tag file, or the
    directory structure, which is not guessable from the prompt.
    """
    answer = (state.get("final_answer") or "").lower()
    if any(v in answer for v in ("villager_job_sites", "job_sites",
                                 "minecraft:beds", "c:chests", "c:barrels")):
        return True, "quoted the values it read"
    if "break_protected" in answer and "medium" in answer:
        return True, "named the tag path from inside the archive"
    if "прислать" in answer or "пришлите" in answer or "вставьте" in answer:
        return False, f"asked the user for the file it already has: {answer[:90]}"
    return False, f"no evidence it opened the jar: {answer[:120]}"


def _check_json_fixed(box, state):
    try:
        text = box.read_text("config/settings.json")
    except Exception as exc:
        return False, f"the file is gone: {exc}"
    try:
        json.loads(text)
    except ValueError as exc:
        return False, f"still invalid: {exc}"
    return True, "the file parses now"



# The REAL mod, when it is on this machine. Kept separate from _two_mods on
# purpose: the synthetic morevillagers ships a tidy
# data/morevillagers/tags/block/villager_job_sites.json listing the block ids,
# and the real one does NOT. Measured on the actual 6.0.0 jar: its only job-site
# tag is point_of_interest_type/acquirable_job_site.json, which lists
# PROFESSIONS (morevillagers:miner, :florist, ...), while the eight workstation
# BLOCK ids appear only in assets/.../lang/en_us.json and the recipe paths. So
# the fixture was easier than reality in exactly the place the task is hard --
# there is no file to find, the ids have to be derived.
_REAL_JAR = (Path(__file__).resolve().parent.parent / "runtime" / "sandboxes"
             / "100000001" / "morevillagers-neoforge-1.21.1-6.0.0.jar")

# The eight workstations, read out of the real jar (see the note above).
_REAL_BLOCKS = ("blueprint_table", "decayed_workbench", "gardening_table",
                "hunting_post", "mining_bench", "oceanography_table",
                "purpur_altar", "woodworking_table")


def _real_mod(box: CS.Sandbox) -> str:
    _thief_jar(box)
    shutil.copyfile(_REAL_JAR, box.root / _REAL_JAR.name)
    return _REAL_JAR.name


def _check_real_mod(box, state):
    answer = (state.get("final_answer") or "").lower()
    named = [b for b in _REAL_BLOCKS if b in answer]
    if len(named) >= 4:
        return True, "named %d of 8: %s" % (len(named), ", ".join(named))
    if named:
        return False, "named only %s: %s" % (", ".join(named), answer[:110])
    if "пришли" in answer or "прислать" in answer or "вставь" in answer:
        return False, "asked for files it already has: " + answer[:110]
    return False, "no workstation named: " + answer[:150]



# ── harder fixtures: real project data, real deliverables ───────────────────

_PERSONALITIES = Path(__file__).resolve().parent.parent / "personalities"


def _real_folder(box: CS.Sandbox) -> str:
    """A COPY of the project's own personalities/ -- 22 real files.

    Real data, not a fixture: hand-made trees are uniform in ways that make
    every question easy. These have mixed .json and .txt, Russian names, and
    casts that differ in size, which is what "look at my folder and tell me
    what is in it" actually looks like.
    """
    shutil.copytree(_PERSONALITIES, box.root / "personalities")
    return "personalities"


def _check_real_folder(box, state):
    answer = (state.get("final_answer") or "").lower()
    casts = [p.stem.replace("cast_", "") for p in _PERSONALITIES.glob("cast_*.json")]
    named = [c for c in casts if c in answer]
    if len(named) >= len(casts) - 1:
        return True, "named %d of %d casts: %s" % (len(named), len(casts),
                                                   ", ".join(named))
    if named:
        return False, "named only %s of %d: %s" % (", ".join(named), len(casts),
                                                   answer[:110])
    return False, "never listed the casts: " + answer[:150]


_INI = """; Mantella runtime settings. Comments matter: they are the only
; documentation of what these numbers were measured to do.
[Speech]
; seconds of silence before the player's turn is considered over
pause_threshold = 0.7
fast_response_mode = 1
; how many words are sent to TTS at once
number_words_tts = 8

[LLM]
; measured: a mismatched context rebuilds the runner on every call
num_ctx = 8192
temperature = 0.85
max_tokens = 250
"""


def _config_ini(box: CS.Sandbox) -> str:
    box.write_text("config/config.ini", _INI)
    return "config/config.ini"


def _check_config_ini(box, state):
    try:
        text = box.read_text("config/config.ini")
    except Exception as exc:
        return False, f"config unreadable after the edit: {exc}"
    import configparser
    cp = configparser.ConfigParser(allow_no_value=True, comment_prefixes=(";",))
    try:
        cp.read_string(text)
    except Exception as exc:
        return False, f"the file no longer parses: {exc}"
    try:
        ctx_ok = cp["LLM"]["num_ctx"].strip() == "16384"
        temp_ok = cp["LLM"]["temperature"].strip() == "0.6"
    except KeyError as exc:
        return False, f"a key was lost: {exc}"
    kept = cp["Speech"].get("pause_threshold", "").strip() == "0.7"
    comments = text.count(";") >= 4
    if not ctx_ok:
        return False, f"num_ctx is {cp['LLM'].get('num_ctx')!r}, not 16384"
    if not temp_ok:
        return False, f"temperature is {cp['LLM'].get('temperature')!r}, not 0.6"
    if not kept:
        return False, "an unrelated key in [Speech] was changed"
    if not comments:
        return False, "the comments were stripped (%d left)" % text.count(";")
    return True, "both values changed, comments and other keys intact"


_CSV = """model,quant,tool_calls_ok,turn_seconds
gemma4-26b,IQ4_XS,18,0.67
gemma4-26b,Q3_K_P,12,0.61
qwen35-hauhau,IQ4_XS,15,11.90
qwen35-hauhau,Q4_K_M,16,13.40
"""


def _chart_data(box: CS.Sandbox) -> str:
    box.write_text("measurements.csv", _CSV)
    return "measurements.csv"


def _check_chart(box, state):
    """A chart is only made if a real PNG exists and the user can receive it."""
    pngs = [p for p in box.root.rglob("*.png") if AGENT_DIR not in p.parts]
    if not pngs:
        # A dead container engine is not a model failure, and scoring it as one
        # accuses the model of something it got RIGHT: measured with Docker
        # stopped, all three runs said "среда выполнения недоступна, запустите
        # Docker" instead of inventing a chart, which is exactly the behaviour
        # this bench exists to protect. Distinguishing them is the difference
        # between a red suite and a missing daemon.
        import code_runner as _cr
        if not _cr.docker_available(timeout=20):
            return False, ("CANNOT RUN: the container engine is not running, so "
                           "no chart was possible. The model declined honestly.")
        return False, "no PNG was produced at all"
    from PIL import Image
    biggest = max(pngs, key=lambda p: p.stat().st_size)
    try:
        with Image.open(biggest) as im:
            w, h = im.size
    except Exception as exc:
        return False, f"{biggest.name} is not a readable image: {exc}"
    if w < 200 or h < 150:
        return False, f"{biggest.name} is {w}x{h}, too small to be a chart"
    shown = str(state.get("image_path") or "")
    if not shown:
        return False, (f"made {biggest.name} ({w}x{h}) but never called "
                       f"open_image, so the user receives nothing")
    return True, f"{Path(shown).name} {w}x{h}, handed over for delivery"


CASES = [
    dict(id="real_folder",
         setup=_real_folder,
         text="В папке personalities лежат мои настоящие файлы. Посмотри и "
              "скажи, какие там есть касты (cast_*.json) — перечисли все.",
         check=_check_real_folder),
    dict(id="config_tune",
         setup=_config_ini,
         text="В config/config.ini поставь num_ctx = 16384 и temperature = 0.6. "
              "Остальное не трогай, комментарии сохрани.",
         check=_check_config_ini),
    dict(id="chart_from_csv",
         setup=_chart_data,
         text="В measurements.csv мои замеры. Построй питоном столбчатый график "
              "turn_seconds по моделям, сохрани картинкой и покажи мне.",
         check=_check_chart),
    dict(id="real_mod",
         setup=_real_mod,
         text="Тут два джарника: Thief и настоящий MoreVillagers. Thief "
              "защищает тег c:villager_job_sites. Посмотри внутрь "
              "MoreVillagers и скажи, какие конкретно блоки-рабочие места он "
              "добавляет — полные id.",
         check=_check_real_mod),
    dict(id="tag_edit",
         setup=_thief_jar,
         text="Вот джарник мода Thief. Распакуй его, найди тег "
              "break_protected/medium и добавь туда блок "
              "morevillagers:trading_table, существующие значения не трогай. "
              "Потом запакуй обратно и пришли мне архив.",
         check=_check_tag_edit),
    dict(id="read_and_answer",
         setup=_thief_jar,
         text="Открой присланный джарник и скажи, какие теги отвечают за "
              "защиту от разрушения средней тяжести.",
         check=_check_read_and_answer),
    dict(id="cross_mod",
         setup=_two_mods,
         text="Тут два джарника: Thief и morevillagers. Thief защищает тег "
              "c:villager_job_sites. Посмотри в обоих и скажи, какие конкретно "
              "блоки из morevillagers попадут под эту защиту.",
         check=_check_cross_mod),
    dict(id="deliver_fix",
         setup=_two_mods,
         # Names WHAT to change and never HOW to hand it back. The first
         # version of this case said only "жители не реагируют — почини это",
         # which made the model diagnose the mod AND deliver the result, so a
         # failure could not tell the two apart: one run packed a correct
         # archive of the WRONG mod, and scored the same as a run that wrote
         # nothing. A case that fails for two reasons measures neither.
         text="Тут два джарника: Thief и MoreVillagers. В Thief тег "
              "break_protected/medium.json перечисляет блоки, которые нельзя "
              "ломать, и рабочих мест из MoreVillagers там нет — поэтому кражу "
              "с них не замечают. Добавь их туда.",
         check=_check_deliver_fix),
    dict(id="deliver_symptom",
         setup=_two_mods,
         # The HARD half, kept separate on purpose. deliver_fix names the file
         # so it measures delivery alone; this one names only the symptom, the
         # way the live chat did, so it measures diagnosis AND delivery
         # together. Scored apart because a single case that can fail for two
         # reasons measures neither -- which is how the first version of this
         # bench blamed the model for a broken fixture.
         text="Тут два джарника: Thief и MoreVillagers. Жители из MoreVillagers "
              "не реагируют на кражу со своих рабочих мест — почини это.",
         check=_check_deliver_fix),
    dict(id="photo_in_folder",
         setup=_photo,
         text="В рабочей папке лежит scan.png. Посмотри, что на нём нарисовано, "
              "и скажи фигуру и цвет.",
         check=_check_photo_read),
    dict(id="fix_broken_json",
         setup=_broken_json,
         text="В config/settings.json ошибка, файл не парсится. Найди её и "
              "почини.",
         check=_check_json_fixed),
]


# -- coding-agent tasks (2026-09-23): outline / run_tests / plan / undo ----------

def _buggy_pkg(box: CS.Sandbox) -> str:
    """A small project whose tests fail for one real bug, hidden in a
    neighbour of the function the traceback names."""
    box.write_text("shop/__init__.py", "")
    box.write_text("shop/money.py",
                   "def to_cents(rub):\n    return int(rub * 100)\n\n\n"
                   "def apply_discount(cents, percent):\n"
                   "    return cents - cents * percent // 10\n")
    box.write_text("shop/cart.py",
                   "from shop.money import to_cents, apply_discount\n\n\n"
                   "class Cart:\n    def __init__(self):\n        self.items = []\n\n"
                   "    def add(self, name, rub):\n        self.items.append((name, to_cents(rub)))\n\n"
                   "    def total(self, percent=0):\n"
                   "        return apply_discount(sum(c for _, c in self.items), percent)\n")
    box.write_text("tests/test_cart.py",
                   "from shop.cart import Cart\n\n\n"
                   "def test_total():\n    c = Cart(); c.add('a', 10); c.add('b', 5)\n"
                   "    assert c.total() == 1500\n\n\n"
                   "def test_discount():\n    c = Cart(); c.add('a', 200)\n"
                   "    assert c.total(10) == 18000\n")
    return "shop"


def _pytest_green(box) -> tuple:
    import code_runner
    res = code_runner.run_python(
        box, "import os, sys, pytest; sys.path.insert(0, os.getcwd()); sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', 'tests']))",
        timeout=300, allow_host=False)
    return res.code == 0, (res.output or "")[-300:]


def _check_fix_with_tests(box, state):
    ok, out = _pytest_green(box)
    if not ok:
        return False, "tests still fail: " + out.replace("\n", " ")[-150:]
    t = box.read_text("tests/test_cart.py")
    if "18000" not in t or "1500" not in t:
        return False, "made the tests pass by editing the tests"
    return True, "tests green, fixed in the code"


def _mod_sources(box: CS.Sandbox) -> str:
    """The real request: an archive of a mod's files, 'build me the jar'. The
    trap is the extra top folder -- a jar with mymod/pack.mcmeta inside is not
    a mod Minecraft will load."""
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mymod/pack.mcmeta", json.dumps({"pack": {"pack_format": 48, "description": "My mod"}}))
        zf.writestr("mymod/META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")
        zf.writestr("mymod/fabric.mod.json", json.dumps({"schemaVersion": 1, "id": "mymod", "version": "1.0"}))
        zf.writestr("mymod/data/mymod/tags/block/soft.json", json.dumps({"values": ["minecraft:dirt"]}))
    (box.root / "mods.zip").write_bytes(buf.getvalue())
    return "mods.zip"


def _check_build_jar(box, state):
    jars = [p for p in box.root.rglob("*.jar") if AGENT_DIR not in p.parts]
    if not jars:
        return False, "no .jar produced"
    for j in jars:
        try:
            names = set(zipfile.ZipFile(j).namelist())
        except zipfile.BadZipFile:
            continue
        if {"pack.mcmeta", "fabric.mod.json"} <= names and \
                "data/mymod/tags/block/soft.json" in names:
            return True, f"{j.name}: mod files at the jar root"
    return False, f"jar(s) {[j.name for j in jars]} lack pack.mcmeta/fabric.mod.json at the root"


def _empty(box: CS.Sandbox) -> str:
    return ""


def _check_utility(box, state):
    import code_runner
    if not (box.root / "slug.py").exists():
        return False, "no slug.py"
    tests = [p for p in box.root.rglob("test_*.py") if AGENT_DIR not in p.parts]
    if not tests:
        return False, "no test file written"
    res = code_runner.run_python(
        box, "from slug import slugify\n"
             "assert slugify('Hello, World!') == 'hello-world', slugify('Hello, World!')\n"
             "assert slugify('  a  b  ') == 'a-b', slugify('  a  b  ')\nprint('ok')",
        timeout=60, allow_host=False)
    if not res.ok:
        return False, "slugify wrong: " + (res.output or "")[-120:].replace("\n", " ")
    ok, out = _pytest_green(box) if (box.root / "tests").is_dir() else (True, "")
    rec = ""
    try:
        rec = (box.root / AGENT_DIR / "record.txt").read_text(encoding="utf-8")
    except Exception:
        pass
    if "run_tests" not in rec and "run_code" not in rec:
        return False, "never ran its tests"
    return True, "utility correct, tests written and run"


CASES += [
    dict(id="fix_with_tests",
         setup=_buggy_pkg,
         text="В проекте shop падают тесты. Найди причину и почини код "
              "(тесты не трогай), потом убедись, что всё зелёное.",
         check=_check_fix_with_tests),
    dict(id="build_jar",
         setup=_mod_sources,
         text="В mods.zip лежат файлы моего мода. Собери из них jar, который "
              "можно положить в папку mods.",
         check=_check_build_jar),
    dict(id="write_utility",
         setup=_empty,
         text="Напиши модуль slug.py с функцией slugify(text): нижний регистр, "
              "всё кроме букв и цифр заменить на дефис, без дефисов по краям и "
              "без двойных. Напиши к нему pytest-тесты и прогони их.",
         check=_check_utility),
]


# ── runner ──────────────────────────────────────────────────────────────────

DUMP_DIR = None     # --dump: the whole turn, to see WHY a run failed


def _dump_trace(case, state, calls, err):
    """Every message of the turn: the model's text, each tool call with its
    arguments, each tool result (trimmed), and the final answer."""
    msgs = []
    for m in state.get("messages") or []:
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else "?")
        content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        tcs = getattr(m, "tool_calls", None) if not isinstance(m, dict) else m.get("tool_calls")
        msgs.append({"role": role, "content": str(content or "")[:3000],
                     "tool_calls": [{"name": (t.get("name") or (t.get("function") or {}).get("name")),
                                     "args": str(t.get("args") or (t.get("function") or {}).get("arguments"))[:1500]}
                                    for t in (tcs or []) if isinstance(t, dict)]})
    out = Path(DUMP_DIR) / f"{case['id']}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"case": case["id"], "task": case["text"], "calls": calls,
                               "error": err, "final_answer": state.get("final_answer"),
                               "messages": msgs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"   trace -> {out}")

def run_case(graph, ctx, case, base: Path):
    box = CS.sandbox_for(f"e2e_{case['id']}_{int(time.time()*1000)}", base=base)
    try:
        case["setup"](box)
    except FileNotFoundError as exc:     # a fixture from the user's real files is gone
        return dict(id=case["id"], ok=None, why=f"SKIPPED: {exc.filename} missing",
                    seconds=0.0, box=box, calls=[])

    ctx.sandbox = box
    ctx.sandbox_user = _User()
    ctx.session_memory.clear()
    ctx.pinned_facts = []
    ctx.last_image_path = None

    state = {"messages": [], "user_input": case["text"], "image_data": None}
    # Which tools actually ran. Wrapping execute_tool rather than reading the
    # transcript: a call the loop rejected before dispatch never appears in the
    # messages, and that is exactly the failure worth seeing.
    calls = []
    _orig = graph.execute_tool

    def _rec(ctx_, state_, name, args):
        # The ARGUMENT matters as much as the name. Two failing runs on the real
        # jar both showed ['search_files' x4, 'list_files' x5] and gave up --
        # a trace that says the model searched, and nothing about WHAT for. The
        # first thing worth knowing is whether it searched for something that
        # exists.
        probe = ""
        if isinstance(args, dict):
            probe = str(args.get("pattern") or args.get("path") or "")[:40]
        calls.append(name + ("(" + probe + ")" if probe else ""))
        return _orig(ctx_, state_, name, args)

    graph.execute_tool = _rec
    t0 = time.perf_counter()
    err = ""
    try:
        state = graph.personality_node(ctx, state)
    except Exception as exc:                 # a crash is a result, not a stop
        err = f"{type(exc).__name__}: {exc}"
    finally:
        graph.execute_tool = _orig
    dt = time.perf_counter() - t0

    if DUMP_DIR:
        _dump_trace(case, state, calls, err)
    if err:
        return dict(id=case["id"], ok=False, why=err, seconds=dt, box=box,
                    calls=calls)
    ok, why = case["check"](box, state)
    return dict(id=case["id"], ok=ok, why=why, seconds=dt, box=box,
                calls=calls,
                answer=(state.get("final_answer") or "")[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--keep", action="store_true",
                    help="keep the sandboxes for inspection")
    ap.add_argument("--dump", default=None, help="write each turn's full trace here")
    args = ap.parse_args()
    global DUMP_DIR
    DUMP_DIR = args.dump

    cases = [c for c in CASES if not args.case or c["id"] in args.case]
    graph, ctx, _img = setup()
    base = Path(tempfile.mkdtemp(prefix="sbx_e2e_"))
    print(f"sandboxes under {base}\n")

    rows = []
    for rep in range(args.reps):
        for case in cases:
            r = run_case(graph, ctx, case, base)
            if r["ok"] is None:
                print(f"[SKIP] {r['id']:<18} {r['why'][:90]}")
                continue
            rows.append(r)
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"[{mark}] {r['id']:<18} {r['seconds']:>5.1f}s  {r['why'][:90]}")
            if not r["ok"]:
                # The answer is the diagnosis: "I cannot open archives" is a
                # different bug from a tool that ran and produced nothing.
                print(f"        tools: {r.get('calls')}")
                print(f"        said : {(r.get('answer') or '')[:200]}")

    print("\n" + "=" * 72)
    for cid in dict.fromkeys(c["id"] for c in cases):
        got = [r for r in rows if r["id"] == cid]
        n = sum(1 for r in got if r["ok"])
        print(f"  {cid:<20} {n}/{len(got)}")
    total = sum(1 for r in rows if r["ok"])
    print("-" * 72)
    print(f"  TOTAL {total}/{len(rows)}")

    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)
    else:
        print(f"\n  sandboxes kept in {base}")
    return 0 if total == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
