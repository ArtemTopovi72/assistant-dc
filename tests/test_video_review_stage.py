from types import SimpleNamespace

import stages
import tool_image_handlers as handlers


def test_join_review_reports_checking_stage():
    seen = []
    def vision(ctx, **kwargs):
        assert seen == ['Checking the video']
        return 'NONE'
    ctx = SimpleNamespace(set_stage=seen.append, vision_model=vision)
    assert handlers._check_joins(ctx, []) == ''
    assert not seen
    assert handlers._check_joins(ctx, [('before.jpg', 'after.jpg')]) == ''
    assert stages.translate('Checking the video', 'ru') == 'Проверяю видео'
