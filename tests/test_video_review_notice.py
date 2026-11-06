from graph_personality import _video_review_notice
from graph_personality import _approved_video_reply
from types import SimpleNamespace


def test_video_review_notice_does_not_report_uncertain_findings_as_facts():
    answer = _video_review_notice('Готово!', 'ball did not move', 'Оживи шар')
    assert 'возможные отличия' in answer
    assert 'ball did not move' not in answer
    assert _video_review_notice('Готово!', '', 'Оживи шар') == 'Готово!'
    assert 'possible differences' in _video_review_notice('Done!', 'uncertain', 'Animate the balloon')


def test_approved_video_reply_does_not_claim_removed_scenes_or_duration():
    ctx = SimpleNamespace(video_plan_approved_result=True, video_plan_warnings=['speech'])
    result = _approved_video_reply(ctx, 'Готово, 25 секунд и чашка!', 'Продолжи видео')
    assert '25' not in result and 'чашка' not in result
    assert 'реч' in result.lower()
    assert _approved_video_reply(SimpleNamespace(), 'Готово', 'Продолжи') == 'Готово'
