from graph_personality import _video_review_notice
from graph_personality import _approved_video_reply
from types import SimpleNamespace
import tool_image_handlers as handlers


def test_graphic_and_inset_warnings_both_reach_user():
    note = ('A person may cross the boundary of an inset without being asked. '
            'The background may have changed unexpectedly. '
            'The video may contain unrequested graphic overlays: Green stars.')
    answer = _video_review_notice('Готово!', note, 'Оживи фото')
    assert 'видеовставки' in answer and 'графические эффекты' in answer
    assert 'возможные отличия' in answer


def test_inset_review_reports_positive_only():
    frames = [(0, 'a.jpg'), (1, 'b.jpg'), (4, 'c.jpg')]
    yes = SimpleNamespace(vision_model=lambda *a, **kw: 'YES: the arm crosses into the studio')
    no = SimpleNamespace(vision_model=lambda *a, **kw: 'NO: inside the inset')
    unknown = SimpleNamespace(vision_model=lambda *a, **kw: 'UNKNOWN')
    assert handlers._check_inset_crossing(yes, frames, ['source.jpg'], 'Take the cup')
    assert not handlers._check_inset_crossing(no, frames, ['source.jpg'], 'Take the cup')
    assert not handlers._check_inset_crossing(unknown, frames, ['source.jpg'], 'Take the cup')


def test_inset_review_samples_across_full_continuation():
    calls = []
    ctx = SimpleNamespace(vision_model=lambda *a, **kw: calls.append(kw) or 'NO')
    frames = [(i, f'{i}.jpg') for i in range(10)]
    handlers._check_inset_crossing(ctx, frames, ['source.jpg'], 'Continue the video')
    paths = [path for call in calls for _, path in call['image_paths']]
    assert {'source.jpg', '0.jpg', '4.jpg', '6.jpg', '9.jpg'} <= set(paths)


def test_inset_review_checks_later_groups_after_a_clean_first_group():
    answers = iter(['NO', 'YES: hand crosses the inset'])
    ctx = SimpleNamespace(vision_model=lambda *a, **kw: next(answers))
    frames = [(i, f'{i}.jpg') for i in range(6)]
    assert handlers._check_inset_crossing(ctx, frames, ['source.jpg'], 'Continue')


def test_inset_review_samples_first_seconds_after_original_clip():
    times = handlers._inset_sample_times(15.796, 9.613)
    assert any(9.8 < t < 10.2 for t in times)
    assert any(10.5 < t < 11.5 for t in times)
    assert max(times) < 15.796


def test_video_review_notice_does_not_report_uncertain_findings_as_facts():
    answer = _video_review_notice('Готово!', 'ball did not move', 'Оживи шар')
    assert 'возможные отличия' in answer
    assert 'ball did not move' not in answer
    assert _video_review_notice('Готово!', '', 'Оживи шар') == 'Готово!'
    assert 'possible differences' in _video_review_notice('Done!', 'uncertain', 'Animate the balloon')


def test_failed_video_review_is_not_reported_as_a_detected_problem():
    answer = _video_review_notice('Готово!', 'The video could not be checked against the request.',
                                  'Оживи фото')
    assert 'Не удалось автоматически проверить' in answer
    assert 'заметила возможные отличия' not in answer
    both = _video_review_notice('Готово!',
        'The video could not be checked against the request. The background may have changed unexpectedly.',
        'Оживи фото')
    assert 'Не удалось автоматически проверить' in both and 'возможные отличия' in both


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


def test_unapproved_video_with_drift_does_not_claim_scene_is_unchanged():
    ctx = SimpleNamespace(video_plan_approved_result=False, video_plan_warnings=['framing'])
    result = _approved_video_reply(ctx, 'Синий фон остался неподвижным.', 'Продолжи видео')
    assert 'Синий фон остался' not in result
    assert 'кадре' in result


def test_video_with_unverified_audio_does_not_claim_sound_was_heard():
    ctx = SimpleNamespace(video_plan_approved_result=False, video_plan_warnings=[],
                          video_audio_unverified=True)
    result = _approved_video_reply(ctx, 'Готово, слышен шум комнаты.', 'Оживи фото')
    assert 'слышен' not in result and 'шум комнаты' not in result


def test_approved_video_reply_reports_incomplete_result():
    ctx = SimpleNamespace(video_plan_approved_result=True, video_plan_warnings=['incomplete'])
    result = _approved_video_reply(ctx, 'Всё готово!', 'Продолжи видео')
    assert 'неполным' in result and 'не создались' not in result


def test_short_video_reply_does_not_claim_requested_length():
    ctx = SimpleNamespace(video_plan_approved_result=True, video_plan_warnings=['short'])
    result = _approved_video_reply(ctx, 'Ролик на 25 секунд готов!', 'Продолжи видео')
    assert 'короче запрошенного' in result and '25' not in result
