"""
Miscellaneous metrics that may be useful building blocks for scoring functions.
"""

from collections.abc import Sequence

from decoding.types import T


def levenshtein(s1: Sequence[T], s2: Sequence[T]) -> int:
    """
    Compute the Levenshtein distance between two sequences.

    Args:
        s1: The first sequence.
        s2: The second sequence.

    Returns:
        The Levenshtein distance between the two sequences.

    Examples:
        ```python
        from decoding.estimators import commutativeMBR
        from decoding.metrics import levenshtein
        from decoding.pmf import CategoricalLogPMF

        s1 = "kitten"
        s2 = "sitting"
        assert levenshtein(s1, s2) == 3

        c = ["car", "can", "cat", "bat", "hat"]
        d = CategoricalLogPMF.from_samples(c)
        samples = commutativeMBR(d, utility=lambda s1, s2: -levenshtein(s1, s2))
        assert samples[0].item == "cat"
        ```

    """
    (s1, s2) = (s2, s1) if len(s1) > len(s2) else (s1, s2)
    dists = list(range(len(s1) + 1))
    for idx2, c2 in enumerate(s2):
        new = [idx2 + 1]
        for idx1, c1 in enumerate(s1):
            if c1 == c2:
                new.append(dists[idx1])
            else:
                new.append(1 + min((dists[idx1], dists[idx1 + 1], new[-1])))
        dists = new
    return dists[-1]
