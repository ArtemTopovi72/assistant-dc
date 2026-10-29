import klein


def test_long_element_description_is_not_repeated_in_position_and_table():
    desc = 'a long specific scene description describing a person in a red shirt'
    caption = {'high_level_description': 'A scene', 'compositional_deconstruction': {
        'elements': [{'desc': desc, 'bbox': [100, 100, 500, 500]},
                     {'desc': 'a sign', 'text': 'СОЛ ВЕРНУЛСЯ', 'bbox': [50, 50, 250, 900]}]}}
    prompt = klein.caption_to_prompt(caption)
    assert prompt.count(desc) == 1
    assert 'СОЛ ВЕРНУЛСЯ' in prompt and '<Ru_Text_1>' in prompt
    assert 'lettering fully inside the frame' in prompt
