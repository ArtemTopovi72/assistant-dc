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
