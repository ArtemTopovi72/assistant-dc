from graph_personality import _video_review_notice
from graph_personality import _approved_video_reply
from types import SimpleNamespace
import tool_image_handlers as handlers


def test_inset_review_reports_positive_only():
    frames = [(0, 'a.jpg'), (1, 'b.jpg'), (4, 'c.jpg')]
    yes = SimpleNamespace(vision_model=lambda *a, **kw: 'YES: the arm crosses into the studio')
    no = SimpleNamespace(vision_model=lambda *a, **kw: 'NO: inside the inset')
    unknown = SimpleNamespace(vision_model=lambda *a, **kw: 'UNKNOWN')
    assert handlers._check_inset_crossing(yes, frames, ['source.jpg'], 'Take the cup')
    assert not handlers._check_inset_crossing(no, frames, ['source.jpg'], 'Take the cup')
    assert not handlers._check_inset_crossing(unknown, frames, ['source.jpg'], 'Take the cup')


def test_video_review_notice_does_not_report_uncertain_findings_as_facts():
    answer = _video_review_notice('Готово!', 'ball did not move', 'Оживи шар')
    assert 'возможные отличия' in answer
    assert 'ball did not move' not in answer
    assert _video_review_notice('Готово!', '', 'Оживи шар') == 'Готово!'
    assert 'possible differences' in _video_review_notice('Done!', 'uncertain', 'Animate the balloon')


def test_inset_review_identifies_the_possible_problem():
    answer = _video_review_notice('Готово!', 'A person may cross the boundary of an inset without being asked.',
                                  'Продолжи видео')
    assert 'возможно' in answer.lower() and 'вставк' in answer.lower()


def test_approved_video_reply_does_not_claim_removed_scenes_or_duration():
    ctx = SimpleNamespace(video_plan_approved_result=True, video_plan_warnings=['speech'])
    result = _approved_video_reply(ctx, 'Готово, 25 секунд и чашка!', 'Продолжи видео')
    assert '25' not in result and 'чашка' not in result
    assert 'реч' in result.lower()
    assert _approved_video_reply(SimpleNamespace(), 'Готово', 'Продолжи') == 'Готово'


def test_approved_video_reply_reports_incomplete_result():
    ctx = SimpleNamespace(video_plan_approved_result=True, video_plan_warnings=['incomplete'])
    result = _approved_video_reply(ctx, 'Всё готово!', 'Продолжи видео')
    assert 'только часть' in result and 'не создались' in result


def test_short_video_reply_does_not_claim_requested_length():
    ctx = SimpleNamespace(video_plan_approved_result=True, video_plan_warnings=['short'])
    result = _approved_video_reply(ctx, 'Ролик на 25 секунд готов!', 'Продолжи видео')
    assert 'короче запрошенного' in result and '25' not in result
