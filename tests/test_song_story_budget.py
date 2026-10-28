import music


def test_length_trim_drops_repeated_choruses_before_story_events():
    verse1 = '[verse]\nРыбак пришёл\nЗабросил нить\nВ туман вошёл\nРешил ловить'
    verse2 = '[verse]\nСапог в руке\nВ нём рыбка ждёт\nБлеск на крючке\nТри чуда шлёт'
    verse3 = '[verse]\nСапог умчал\nЗа ним гонясь\nРыбак отстал\nВ траву валясь'
    chorus = '[chorus]\nВот чудеса\nИ смех и стыд\nЗа полчаса\nСапог летит'
    outro = '[outro]\nОсталась история\nСтепан рассмеялся'
    text = '\n\n'.join([verse1, chorus, verse2, chorus, verse3, chorus, outro])
    result = music.trim_to_lines(text, 18)
    assert music.count_sung_lines(result) <= 18
    assert verse1 in result and verse2 in result and verse3 in result
    assert result.endswith(outro) and result.count('[chorus]') == 1


# A 30-second request reached YuE2 as fourteen lines (~80 seconds): the old
# trimming loop stopped at three sections even when those still exceeded the cap.
def test_short_song_compacts_the_whole_story_instead_of_dropping_events():
    import json
    from types import SimpleNamespace
    source = ('[verse]\nCat searched the grass.\nFound the missing key.\n'
              'Carried it to the door.\nThe rain began to pour.\n\n'
              '[chorus]\nCat turned the key.\nOpened the door.\n'
              'The room was warm.\nHe left the storm.\n\n'
              '[outro]\nCat went home to his owner.\nShe hugged him at the door.')
    compact = ('[verse]\nCat found the missing key.\nHe opened the door with glee.\n'
               '[outro]\nBack home to his owner he ran.\nShe hugged her brave little man.')
    calls = []
    def model(ctx, system, user, **kw):
        calls.append(user)
        return json.dumps({'lyrics': source if len(calls) == 1 else compact,
                           'style': 'Global Metadata: pop. Vocal Details: male. Arrangement: guitar.'})
    ctx = SimpleNamespace(text_model=model, yes_no=lambda *a, **k: False)
    result = music.build_structured_caption(ctx, 'Cat finds key, opens door, returns home to owner',
                                            'en', duration_s=30)
    assert music.count_sung_lines(result['lyrics']) <= 6
    assert 'found the missing key' in result['lyrics']
    assert 'opened the door' in result['lyrics']
    assert 'home to his owner' in result['lyrics']
    assert source in calls[-1]


def test_uncooperative_compactor_cannot_ship_an_overbudget_song():
    import json
    from types import SimpleNamespace
    import pytest
    calls = []
    long = '[verse]\n' + 'word\n' * 20
    def model(*a, **k):
        calls.append(1)
        return json.dumps({'lyrics': long, 'style': 'Global Metadata: pop. Vocal Details: male. Arrangement: guitar.'})
    ctx = SimpleNamespace(text_model=model, yes_no=lambda *a, **k: False)
    with pytest.raises(music.SongwritingFailed):
        music.build_structured_caption(ctx, 'a short story', 'en', duration_s=30)
    assert len(calls) <= 3
