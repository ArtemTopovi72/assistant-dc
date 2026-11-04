from types import SimpleNamespace

import tool_image_handlers as handlers


def test_numbered_redraw_uses_original_album_after_previous_edit(monkeypatch):
    monkeypatch.setattr(handlers, '_typography_only', lambda *a: False)
    monkeypatch.setattr(handlers, '_undo', lambda *a: None)
    deps = SimpleNamespace(exists=lambda path: True,
                           _render_budget_exhausted=lambda *a: 'STOP')
    ctx = SimpleNamespace(last_image_path='previous_result.png',
                          recent_image_paths=['first.jpg', 'second.jpg'], deps=deps)
    state = {'image_path': 'previous_result.png', 'edit_before': 'second.jpg'}
    assert handlers._handle_redraw_image(ctx, state, {'instructions': 'Turn picture 1 into anime'}) == 'STOP'
    assert state['last_edit_source'] == state['edit_before'] == state['image_path'] == 'first.jpg'
    assert ctx.last_image_path == 'first.jpg'


def test_missing_numbered_album_source_is_not_silently_replaced():
    ctx = SimpleNamespace(last_image_path='previous_result.png',
                          recent_image_paths=['first.jpg', 'second.jpg'],
                          deps=SimpleNamespace(exists=lambda path: True))
    result = handlers._handle_redraw_image(ctx, {}, {'instructions': 'Turn picture 3 into anime'})
    assert result.startswith('[TOOL ERROR]')
