from scripts.heuristics import longest_answer_idx


def test_longest():
    ch = ["a", "bbbb", "ccc"]
    assert longest_answer_idx(ch) == 1

