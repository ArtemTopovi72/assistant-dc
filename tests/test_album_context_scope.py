from graph_personality import _scope_album_outbound
import threading
from pathlib import Path
from models import Context
from graph_compose import _compose_user_message


def test_album_item_uses_current_turn_and_tool_results_without_erasing_history():
    system = {'role': 'system', 'content': 'rules'}
    previous = {'role': 'assistant', 'content': 'Previous photo had two girls'}
    current = {'role': 'user', 'content': 'Restyle the cat'}
    tool = {'role': 'tool', 'content': 'The cat became white'}
    history = [system, previous, current, tool]
    assert _scope_album_outbound(history, current) == [system, current, tool]
    assert history == [system, previous, current, tool]


def test_album_item_keeps_current_image_summary_but_not_sibling_memory():
    ctx = Context(models=None, transcription_cache={}, cache_file=Path('tests/_album_scope.json'),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(), no_think=True)
    ctx.remember('assistant', 'Previous photo had two girls')
    state = {'user_input': '[style_preset] redraw [album_restyle_item]',
             'vision_summary': 'A cat in a cup', 'messages': []}
    text, _, _ = _compose_user_message(ctx, state)
    assert 'A cat in a cup' in text
    assert 'Previous photo had two girls' not in text
