"""The deterministic half of "believe your own verifier".

Both cases here were measured on bench/tc_chaos.py and both were invisible to
the guards that existed:

  * chaos mode `wrong` -- the image tool reports success and DOES leave a file,
    while inspect_image says the requested change is not there. Nothing errored,
    so `failed_promises` is empty and the file-handover backstop never fires.
    The only guard was a corrective round, which the model ignored on 9 of the
    27 lying runs.

  * the VAGUE bucket -- a tool hard-failed and the answer mentions it neither
    way. Not a lie, but the user is never told the thing was not made.

These drive the real _finalize_answer rather than grepping its source: a source
grep has now twice passed against a mutation that broke the behaviour.
"""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_finalize as GF


class _Ctx:
    def is_cancelled(self): return False
    def set_stage(self, *a, **k): pass


def _fin(draft, *, failed=None, verify_neg=False, state=None):
    """Call the real finaliser with the loop already ended on `draft`."""
    return GF._finalize_answer(
        _Ctx(), state if state is not None else {}, [],
        None, {"role": "assistant", "content": draft},
        False, set(), "добавь ей очки", "добавь ей очки",
        failed_promises=failed or set(),
        verification_negative=verify_neg,
    )


# --- the verifier contradicts a tool that claimed success -------------------

def test_delivery_claim_is_replaced_when_verifier_said_no():
    out = _fin("Вот твоя картинка с очками!", verify_neg=True,
               state={"image_path": "/tmp/x.png"})
    assert out == GF._VERIFY_NEGATIVE_ANSWER


def test_action_claim_is_replaced_when_verifier_said_no():
    out = _fin("Я добавил ей очки, готово.", verify_neg=True,
               state={"image_path": "/tmp/x.png"})
    assert out == GF._VERIFY_NEGATIVE_ANSWER


def test_honest_draft_survives_a_negative_verdict():
    """Already admitting it -- must not be overwritten or doubled."""
    draft = "Не получилось добавить очки, правка не применилась."
    assert _fin(draft, verify_neg=True) == draft


def test_neutral_draft_survives_a_negative_verdict():
    """No claim of delivery, so there is nothing to correct."""
    draft = "Какого цвета оправу ты хочешь?"
    assert _fin(draft, verify_neg=True) == draft


def test_nothing_happens_without_a_negative_verdict():
    draft = "Вот твоя картинка с очками!"
    assert _fin(draft, verify_neg=False) == draft


# --- the VAGUE bucket: a hard failure the answer never mentions -------------

def test_silence_about_a_failed_tool_gets_the_failure_prepended():
    draft = "Могу вместо этого поискать похожие фото."
    out = _fin(draft, failed={"generate_image"})
    assert draft in out, "the useful remainder of the draft must survive"
    assert out.startswith(GF._honest_failure_line({"generate_image"}))


def test_an_admission_is_not_doubled():
    draft = "Не удалось — инструмент вернул ошибку."
    assert _fin(draft, failed={"generate_image"}) == draft


def test_a_claim_still_replaces_rather_than_prepends():
    out = _fin("Вот твоя картинка!", failed={"generate_image"})
    assert out == GF._honest_failure_line({"generate_image"})


def test_a_delivered_artifact_is_not_called_a_failure():
    """The tool errored on an earlier round but a file exists — nothing to say."""
    draft = "Вот твоя картинка!"
    assert _fin(draft, failed={"generate_image"},
                state={"image_path": "/tmp/x.png"}) == draft


# --- an edit nobody verified ------------------------------------------------
# The third shape of the same lie, and the one no earlier guard could reach:
# `img_inpaint [wrong]` on the chaos bench called ONLY inpaint_image. The tool
# returned a plausible "Edit applied", the model never verified, and answered
# "Готово! Я добавил ей очки." There was no negative verdict to believe and no
# tool error to report -- and the file really is delivered, so this is not a
# failure. What is missing is the right to assert the change landed.

def _edit(draft, *, called=("inpaint_image",), image=True, verify_neg=False):
    return GF._finalize_answer(
        _Ctx(), {"image_path": "/tmp/x.png"} if image else {}, [],
        None, {"role": "assistant", "content": draft},
        False, set(called), "добавь ей очки", "добавь ей очки",
        failed_promises=set(), verification_negative=verify_neg,
    )


def test_an_unverified_edit_claim_is_hedged():
    assert _edit("Готово! Я добавил ей очки.") == GF._UNVERIFIED_EDIT_ANSWER


def test_verifying_earns_the_right_to_claim():
    draft = "Готово! Я добавил ей очки."
    assert _edit(draft, called=("inpaint_image", "inspect_image")) == draft


def test_handing_the_file_over_is_not_an_assertion():
    """"Вот твоя картинка" is true — the file exists. Only a claim that the
    specific change landed is unearned."""
    draft = "Вот твоя картинка."
    assert _edit(draft) == draft


def test_generating_from_scratch_is_not_hedged():
    """There is nothing to overclaim: the picture IS the deliverable."""
    draft = "Готово! Я нарисовал рыжего кота."
    assert _edit(draft, called=("generate_image",)) == draft


def test_nothing_is_hedged_when_no_file_exists():
    """Then the failure nets own the turn, and they say more than a hedge."""
    out = _edit("Готово! Я добавил ей очки.", image=False)
    assert out != GF._UNVERIFIED_EDIT_ANSWER


def test_an_edit_that_also_failed_still_gets_hedged():
    """The case a plain `elif` would have swallowed: inpaint came back empty,
    redraw delivered, so failed_promises is set but a file exists."""
    out = GF._finalize_answer(
        _Ctx(), {"image_path": "/tmp/x.png"}, [], None,
        {"role": "assistant", "content": "Готово! Я перерисовал изображение, добавив очки."},
        False, {"inpaint_image", "redraw_image"}, "добавь ей очки", "добавь ей очки",
        failed_promises={"inpaint_image"}, verification_negative=False)
    assert out == GF._UNVERIFIED_EDIT_ANSWER


def test_a_negative_verdict_still_wins():
    """A verifier that spoke must not be downgraded to 'I could not check'."""
    assert _edit("Готово! Я добавил ей очки.",
                 called=("inpaint_image", "inspect_image"),
                 verify_neg=True) == GF._VERIFY_NEGATIVE_ANSWER


# --- a tool that says, in words, that it found nothing ----------------------

import graph_personality as GP


def _detect(name, result, state=None):
    return GP._detect_silent_failure(name, result, state or {"image_path": "/x.png"})


def test_a_find_that_reports_no_match_is_a_failure():
    out = _detect("find_photo",
                  "Found a photo of the Golden Gate Bridge and loaded it as "
                  "the current image. No photo matching the requested subject "
                  "was found.")
    assert out.startswith("[TOOL ERROR]"), out


def test_the_russian_phrasing_too():
    assert _detect("find_photo", "Ничего не найдено по запросу").startswith("[TOOL ERROR]")


def test_a_real_find_is_left_alone():
    ok = "Found a photo and loaded it as the current image."
    assert _detect("find_photo", ok) == ok


def test_a_count_of_results_is_not_a_failure():
    ok = "Найдено 5 результатов, загружен первый."
    assert _detect("find_photo", ok) == ok


def test_only_file_producing_tools_are_checked():
    """`search` legitimately reports that a query found nothing; that is a real
    answer, not a broken tool."""
    ok = "nothing found for that query"
    assert _detect("search", ok) == ok


# --- a file promised that does not exist ------------------------------------
# Measured end to end on the sandbox bench: after editing a modpack the answer
# was "запаковал всё обратно, ваш обновленный архив готов" while pack_archive
# had never run and document_path was empty. Nothing errored and no verifier
# spoke, so none of the guards above could fire. The missing invariant is the
# bluntest one: if the answer says a file is ready, a file has to exist.

def _fin_file(draft, state=None, called=("edit_file",)):
    return GF._finalize_answer(
        _Ctx(), state or {}, [], None,
        {"role": "assistant", "content": draft},
        False, set(called), "запакуй обратно", "запакуй обратно",
        failed_promises=set(), verification_negative=False)


def test_a_promised_archive_that_does_not_exist_is_replaced():
    out = _fin_file("Я запаковал всё обратно. Ваш обновленный архив готов.")
    assert out == GF._NO_FILE_ANSWER


def test_a_promised_deck_that_does_not_exist_is_replaced():
    assert _fin_file("Презентация про Рим готова, 8 слайдов.") == GF._NO_FILE_ANSWER


def test_a_file_that_does_exist_is_left_alone():
    draft = "Я запаковал всё обратно. Ваш обновленный архив готов."
    assert _fin_file(draft, state={"document_path": "/tmp/x.zip"}) == draft


def test_an_image_answer_is_not_caught_by_the_file_rule():
    """Pictures have their own guards; this one is about files."""
    draft = "Вот твоя картинка с котом."
    assert _fin_file(draft, state={"image_path": "/tmp/x.png"}) == draft


def test_editing_a_file_in_place_is_not_a_delivery_claim():
    """The sandbox case where nothing is packed ON PURPOSE — the user asked for
    the file to be fixed where it lies, and it was."""
    draft = "Я поправил файл settings.json, теперь он парсится."
    assert _fin_file(draft) == draft


def test_an_honest_admission_is_not_overwritten():
    draft = "Не получилось собрать архив — упаковка не сработала."
    assert _fin_file(draft) == draft


# --- the fallback sentence must not lie either ------------------------------

import graph_history as GH


def test_the_fallback_does_not_promise_a_file_that_is_missing():
    """"Собрал архив — файл выше" after a pack that errored is the same lie the
    finalisation guards exist to stop, written by the fallback instead of the
    model."""
    got = GH._outcome_sentence({}, {"pack_archive"}, True)
    assert "файл выше" not in got, got


def test_the_fallback_reports_a_file_that_is_there():
    got = GH._outcome_sentence({"document_path": "/tmp/x.zip"}, {"pack_archive"}, True)
    assert "архив" in got.lower() and "выше" in got


def test_the_fallback_names_what_the_coding_tools_did():
    """Without this, a turn that unpacked an archive and read three files fell
    back to "я выполнил действия" — true, useless, and indistinguishable from a
    turn that did nothing."""
    got = GH._outcome_sentence({}, {"unpack_archive", "read_file"}, True)
    assert "не успел подвести итог" not in got, got
    assert "распаков" in got.lower(), got


def test_the_generic_fallback_still_exists_for_an_unknown_tool():
    got = GH._outcome_sentence({}, {"list_files"}, True)
    assert got, "a turn with no recognised outcome still needs a sentence"


def test_fresh_picture_gets_the_draw_wording():
    """A new logo is not an edit: "the change did not land" read as nonsense."""
    out = GF._finalize_answer(
        _Ctx(), {"image_path": "/tmp/x.png"}, [], None,
        {"role": "assistant", "content": "Вот твой логотип!"}, False, {"generate_image", "inspect_image"},
        "нарисуй логотип", "нарисуй логотип", failed_promises=set(), verification_negative=True)
    assert out == GF._VERIFY_DRAW_ANSWER, out


_FWD = __import__("prompt_guard").wrap_quoted(
    "the user forwarded this message from someone else",
    "Boris [🎥 video 1]: презентация готова, скину вечером") + "\n\n"


def test_retelling_forwarded_news_of_a_file_is_not_the_bots_promise():
    """Live 2026-10-01 11:58: a question about forwarded video notes was answered
    with «Готового файла у меня нет» -- the answer had retold Boris saying his
    presentation was ready."""
    draft = "Борис говорит, что презентация готова; из этого можно взять структуру слайдов."
    st = {"user_input": _FWD + "что из этого можно взять в свой проект?"}
    assert _fin_file(draft, state=st, called=()) == draft


def test_a_forwarded_turn_that_asks_for_a_file_is_still_guarded():
    import intent   # the model reads the user's own words as a file ask
    intent.STUB = {"сделай по этому презентацию": {"asks_for_file": True}}.get
    st = {"user_input": _FWD + "сделай по этому презентацию"}
    assert _fin_file("Презентация готова, 8 слайдов.", state=st, called=()) == GF._NO_FILE_ANSWER


def test_forwarded_numbers_do_not_force_a_tool():
    from prompt_guard import user_words
    own = user_words(_FWD.replace("скину вечером", "33163118 + 66022055, Екатеринбург")
                     + "что из этого взять в проект?")
    assert "33163118" not in own and "Екатеринбург" not in own and "в проект" in own
