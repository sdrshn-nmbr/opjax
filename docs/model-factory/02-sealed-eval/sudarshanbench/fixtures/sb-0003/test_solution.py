from solution import is_palindrome

def test_palindrome():
    assert is_palindrome('abba')
    assert not is_palindrome('abc')
