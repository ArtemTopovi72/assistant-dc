from types import SimpleNamespace

import prompt_guard
import tg_links
import tg_reply_shape


def test_previous_article_is_not_used_for_question_about_quoted_answer():
    calls = []
    reads = SimpleNamespace(yes_no=lambda question, text, default=False: calls.append(text) or False)
    assert not tg_links.is_page_followup(
        'А как же точки?', 'Пассивной агрессии нет.',
        'Предыдущий ответ анализировал текст, а не статью.',
        'https://example.org/security', ctx=reads)
    assert calls and 'А как же точки?' in calls[0]


def test_previous_article_can_be_used_for_its_own_followup():
    reads = SimpleNamespace(yes_no=lambda question, text, default=False: True)
    assert tg_links.is_page_followup('А когда основан?', '', 'Казань — столица Татарстана.',
                                    'https://example.org/kazan', ctx=reads)


def test_quoted_scaffold_is_not_reported_as_a_page_read_now():
    note = "[The user's message contains a link. Here is that page, read for you -- title: Old (https://example.org/old). Answer from it.]"
    quoted = prompt_guard.wrap_quoted('the user is replying to this message', note)
    assert tg_reply_shape.links_read(quoted + '\nА как же точки?') == []
    assert tg_reply_shape.links_read(quoted + '\n' + note)[0]['url'] == 'https://example.org/old'


def test_uncited_pages_are_not_guessed_from_titles():
    sources = [{'domain': 'example.org', 'title': 'Unrelated page', 'url': 'https://example.org/unrelated'}]
    ctx = SimpleNamespace(text_model=lambda *args, **kwargs: '1')
    result = tg_reply_shape.shape_search_reply('Это похоже на сарказм.', ['тон переписки'], sources, ctx=ctx)
    assert 'https://example.org/unrelated' not in result


def test_failed_search_does_not_register_unusable_sources_or_memory():
    import tools
    saved = []
    existing = {'url': 'https://example.org/earlier'}
    ctx = SimpleNamespace(turn_sources=[existing], turn_queries=[], web_search_enabled=True,
                          set_stage=lambda *_: None, remember=lambda *args: saved.append(args))
    def search(c, query):
        c.turn_sources.append({'url': 'https://example.org/failed'})
        return 'The truncated search result cuts off mid'
    ctx.deps = SimpleNamespace(run_web_search=search)
    result = tools._handle_search(ctx, {'user_input': 'query'}, {'query': 'query'})
    assert result.startswith('[TOOL ERROR]')
    assert ctx.turn_sources == [existing] and not saved


def test_search_exception_does_not_leave_sources():
    import pytest
    import tools
    ctx = SimpleNamespace(turn_sources=[], web_search_enabled=True, set_stage=lambda *_: None,
                          remember=lambda *args: None)
    def broken(c, query):
        c.turn_sources.append({'url': 'https://example.org/failed'})
        raise RuntimeError('search broke')
    ctx.deps = SimpleNamespace(run_web_search=broken)
    with pytest.raises(RuntimeError):
        tools._handle_search(ctx, {'user_input': 'query'}, {'query': 'query'})
    assert not ctx.turn_sources
