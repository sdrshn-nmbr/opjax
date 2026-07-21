from solution import search_insert


def test_found():
    assert search_insert([1, 3, 5, 6], 5) == 2


def test_insert_middle():
    assert search_insert([1, 3, 5, 6], 2) == 1


def test_insert_end():
    assert search_insert([1, 3, 5, 6], 7) == 4


def test_insert_start():
    assert search_insert([1, 3, 5, 6], 0) == 0
