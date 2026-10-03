"""turn_audit: side-effect claims need a successful tool call; failed turns
are kept for triage, complaints file the previous turn, tests never write
into the live folder."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import turn_audit as TA
# the model's read is stubbed; phrases run live in bench/intent_rest_live.py
import intent
_COMPLAINTS = {"нет, не то", "где файл?", "опять не пришло", "я же просил без сахара",
               "нет, не то, я просил электрический"}
intent.STUB = lambda t: {"complaint": t.strip() in _COMPLAINTS}


class _Ctx:
    def __init__(self, owner="42"):
        self.ozon_owner = owner
        self.model_name = "gemma"
        self.no_think = True


@pytest.fixture
def traces(tmp_path, monkeypatch):
    monkeypatch.setattr(TA, "TRACE_DIR", tmp_path)
    monkeypatch.setenv("FAILED_TURNS_DIR", str(tmp_path))
    TA._LAST.clear()
    return tmp_path


@pytest.mark.parametrize("answer,tools,box,want", [
    ("Готово, добавил чайник в корзину.", set(), False, ["cart"]),
    ("Готово, добавил чайник в корзину.", {"ozon_cart"}, False, []),
    ("Хорошо, запомнил: ты живёшь в Казани.", set(), False, ["remember"]),
    ("Хорошо, запомнил.", {"remember_fact"}, False, []),
    ("Поставил пункт выдачи на Ленина, 5.", set(), False, ["location"]),
    ("Запустил тесты — все прошли.", {"edit_file"}, True, ["ran_code"]),
    ("Запустил тесты — все прошли.", set(), False, ["ran_code"]),  # no sandbox: nothing CAN have run
    ("Исправил файл medium.json.", set(), True, ["edited_file"]),
    ("Исправил файл medium.json.", {"edit_file"}, True, []),
    ("Не смог добавить в корзину: Ozon не ответил.", set(), False, []),
    ("Вот три чайника с хорошими отзывами.", set(), False, []),
])
def test_claims(answer, tools, box, want):
    assert TA.unmet_claims(answer, tools, box) == want


def test_an_unmet_claim_gets_one_honest_line():
    out, ids = TA.correct_claims("Добавил в корзину.", set(), False)
    assert ids == ["cart"] and out.startswith("Добавил в корзину.")
    assert "ничего не добавлял" in out


def test_a_failed_turn_is_saved(traces):
    path = TA.finish_turn(_Ctx(), {}, [{"role": "user", "content": "x"}], "x", "",
                          ["empty_answer"], ["search!"])
    d = json.loads(open(path, encoding="utf-8").read())
    assert d["reasons"] == ["empty_answer"] and d["calls"] == ["search!"]


def test_a_clean_turn_is_not_saved_but_a_complaint_files_it(traces):
    assert TA.finish_turn(_Ctx(), {}, [], "найди чайник", "Вот чайник.", [], ["ozon_search"]) == ""
    assert list(traces.glob("*.json")) == []
    path = TA.file_complaint(_Ctx(), "нет, не то, я просил электрический")
    d = json.loads(open(path, encoding="utf-8").read())
    assert "user_complaint" in d["reasons"] and d["user_input"] == "найди чайник"


def test_complaints_are_per_conversation(traces):
    TA.finish_turn(_Ctx("1"), {}, [], "a", "b", [], [])
    assert TA.file_complaint(_Ctx("2"), "не то") == ""


@pytest.mark.parametrize("text,want", [
    ("нет, не то", True), ("где файл?", True), ("опять не пришло", True),
    ("я же просил без сахара", True), ("найди чайник", False),
    ("нету ли дешевле?", False), ("спасибо, отлично", False),
])
def test_complaint_detection(text, want):
    assert TA.is_complaint(text) is want


def test_suites_do_not_write_into_the_live_folder(monkeypatch):
    monkeypatch.delenv("FAILED_TURNS_DIR", raising=False)
    assert TA._enabled() is False


# -- tool_next_step --------------------------------------------------------
import tool_next_step as NS


def test_an_edit_inside_an_unpacked_mod_says_pack_it():
    h = NS.next_step("write_file", {"path": "thief_unpacked/data/a.json"}, "Wrote x.")
    assert "pack_archive 'thief_unpacked'" in h


def test_an_edited_script_says_run_it():
    assert "run_code" in NS.next_step("edit_file", {"path": "src/app.py"}, "Edited src/app.py.")


def test_a_result_with_its_own_guidance_is_left_alone():
    assert NS.next_step("write_file", {"path": "m_unpacked/a"}, "Nothing changed: ...") == ""
    assert NS.next_step("write_file", {"path": "m_unpacked/a"}, "Wrote.\n[NOTE] last round") == ""


def test_an_empty_search_says_do_not_invent():
    assert "not found" in NS.next_step("rag_search", {}, "No stored passages match 'x'.")


def test_an_unknown_tool_names_the_close_ones():
    out = NS.unknown_tool("serach", ["search", "calculate"])
    assert out.startswith("[TOOL ERROR]") and "search" in out


def test_reminder_and_wrote_down_claims():
    assert TA.unmet_claims("Хорошо, я записал. Одна минута до воды!", set(), False) == ["wrote_down"]
    assert TA.unmet_claims("Хорошо, напомню через минуту.", set(), False) == ["reminder"]
    assert TA.unmet_claims("Хорошо, напомню через минуту.", {"set_reminder"}, False) == []
    assert TA.unmet_claims("Не напомню, такой функции нет.", set(), False) == []
    assert TA.unmet_claims("Он записал альбом в 1999.", set(), False) == []


def test_calc_date_helpers_are_results():
    import graph_personality as _gp
    assert _gp._is_calc_result("Thursday") and _gp._is_calc_result("1799-06-06")
    assert not _gp._is_calc_result("1. Beekeeping in temperate climates")
