from solution import balanced

def test_bal():
    assert balanced('(())')
    assert not balanced('(()')
