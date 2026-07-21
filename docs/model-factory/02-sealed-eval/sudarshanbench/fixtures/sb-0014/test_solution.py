from solution import roman_to_int


def test_basic():
    assert roman_to_int("III") == 3
    assert roman_to_int("LVIII") == 58


def test_subtractive():
    assert roman_to_int("IV") == 4
    assert roman_to_int("IX") == 9
    assert roman_to_int("MCMXCIV") == 1994
