import video


def test_spoken_line_does_not_become_a_caption_when_classifier_is_wrong(monkeypatch):
    video._ONSCREEN_CACHE.clear()
    monkeypatch.setattr(video, 'text_model', lambda ctx: lambda *a, **kw: '{"needs":true,"words":["ПРЯМО КАК ВО ВТОРОМ ТИМ ФОРТРЕСЕ"]}')
    assert video._onscreen_text_words(None, 'Персонаж говорит на русском не останавливаясь "Прямо как во втором Тим Фортресе"') == []


def test_explicit_sign_and_spoken_line_are_distinguished(monkeypatch):
    video._ONSCREEN_CACHE.clear()
    monkeypatch.setattr(video, 'text_model', lambda ctx: lambda *a, **kw: '{"needs":true,"words":["ПРИВЕТ","МАГАЗИН"]}')
    assert video._onscreen_text_words(None, 'Он говорит "Привет", а на вывеске написано "МАГАЗИН"') == ['МАГАЗИН']


def test_explicit_subtitles_keep_spoken_line(monkeypatch):
    video._ONSCREEN_CACHE.clear()
    monkeypatch.setattr(video, 'text_model', lambda ctx: lambda *a, **kw: '{"needs":true,"words":["ПРИВЕТ"]}')
    assert video._onscreen_text_words(None, 'Он говорит "Привет", добавь субтитры с этой фразой') == ['ПРИВЕТ']


def test_unquoted_spoken_line_is_not_captioned(monkeypatch):
    video._ONSCREEN_CACHE.clear()
    monkeypatch.setattr(video, 'text_model', lambda ctx: lambda *a, **kw: '{"needs":true,"words":["ПРИВЕТ"]}')
    assert video._onscreen_text_words(None, 'Персонаж говорит Привет и машет рукой') == []


def test_same_line_explicitly_written_without_quotes_is_kept(monkeypatch):
    video._ONSCREEN_CACHE.clear()
    monkeypatch.setattr(video, 'text_model', lambda ctx: lambda *a, **kw: '{"needs":true,"words":["ПРИВЕТ"]}')
    assert video._onscreen_text_words(None, 'Он говорит "Привет", а на вывеске написано ПРИВЕТ') == ['ПРИВЕТ']


def test_do_not_add_caption_when_subtitles_explicitly_disabled(monkeypatch):
    video._ONSCREEN_CACHE.clear()
    monkeypatch.setattr(video, 'text_model', lambda ctx: lambda *a, **kw: '{"needs":true,"words":["ПРИВЕТ"]}')
    assert video._onscreen_text_words(None, 'Он говорит "Привет", без субтитров') == []
