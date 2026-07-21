from solution import group_anagrams


def _normalize(groups):
    return sorted(sorted(g) for g in groups)


def test_groups():
    got = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    assert _normalize(got) == _normalize(
        [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    )


def test_singles():
    assert _normalize(group_anagrams(["a"])) == [["a"]]
