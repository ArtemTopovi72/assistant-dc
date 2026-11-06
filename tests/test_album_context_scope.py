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


def test_video_and_picture_keep_current_evidence_without_old_chat_memory():
    ctx = Context(models=None, transcription_cache={}, cache_file=Path('tests/_album_scope.json'),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(), no_think=True)
    ctx.remember('assistant', 'Old backpack recommendation')
    state = {'user_input': '[video_link_context] Compare these. Video: a cat jumps.',
             'vision_summary': 'A cat on a sofa', 'image_data': True, 'messages': []}
    text, _, _ = _compose_user_message(ctx, state)
    assert 'A cat on a sofa' in text and 'a cat jumps' in text
    assert 'Old backpack recommendation' not in text
    assert 'do not resume earlier tasks or edit the image' in text
