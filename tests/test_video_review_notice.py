from graph_personality import _video_review_notice


def test_video_review_notice_does_not_report_uncertain_findings_as_facts():
    answer = _video_review_notice('Готово!', 'ball did not move', 'Оживи шар')
    assert 'возможные отличия' in answer
    assert 'ball did not move' not in answer
    assert _video_review_notice('Готово!', '', 'Оживи шар') == 'Готово!'
    assert 'possible differences' in _video_review_notice('Done!', 'uncertain', 'Animate the balloon')
