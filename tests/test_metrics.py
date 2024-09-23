import string

from decoding.metrics import levenshtein


def test_levenshtein() -> None:
    s1 = ""
    s2 = "abc"
    assert levenshtein(s1, s2) == levenshtein(s2, s1) == len(s2)

    s1 = "def"
    assert levenshtein(s1, s2) == levenshtein(s2, s1) == len(s2)

    s1 = string.printable
    s2 = s1[::-1]
    assert len(s1) % 2 == 0
    assert levenshtein(s1, s2) == levenshtein(s2, s1) == len(s1)

    s3 = (1, 2, 3)
    s4 = (3, 2, 1)
    assert levenshtein(s3, s4) == levenshtein(s4, s3) == len(s3) - 1

    s5 = list(s3)
    s6 = list(s4)
    assert levenshtein(s5, s6) == levenshtein(s6, s5) == len(s5) - 1
