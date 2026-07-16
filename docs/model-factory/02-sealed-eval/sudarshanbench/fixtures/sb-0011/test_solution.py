from solution import deep_get

def test_deep():
    assert deep_get({'a': {'b': 1}}, 'a.b') == 1
