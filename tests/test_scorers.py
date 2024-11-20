import time
from collections.abc import Sequence

import jax.numpy as jnp

from decoding.pmf import LogPMF, ScoredItem
from decoding.scorers import Scorer


def test_scorer_from_f_str_to_num() -> None:
    def f(s: str) -> int:
        time.sleep(1e-2)
        return len(s)

    d = LogPMF.from_samples(["a", "bb", "ccc"])

    scorer = Scorer.from_f_str_to_num(f)
    t1 = time.time()
    samples = scorer(d)
    t1 = time.time() - t1
    assert [s.score for s in samples] == [3, 2, 1]

    scorer = Scorer.from_f_str_to_num(f, parallelize=True)
    t2 = time.time()
    samples = scorer(d)
    t2 = time.time() - t2
    assert [s.score for s in samples] == [3, 2, 1]

    max_time = 3e-2
    assert t2 < max_time <= t1


def test_scorer_from_f_batch_str_to_batch_num() -> None:
    def f(ss: Sequence[str]) -> list[int]:
        return [len(s) for s in ss]

    d = LogPMF.from_samples(["a", "bb", "ccc"])

    scorer = Scorer.from_f_batch_str_to_batch_num(f)
    samples = scorer(d)
    assert [s.score for s in samples] == [3, 2, 1]


def test_scorer_from_f_logpmf_to_batch_num() -> None:
    def f(d: LogPMF[str]) -> list[float]:
        return [float(jnp.exp(logp) * len(item)) for logp, item in d]

    d = LogPMF.from_samples(["a", "bb", "bb", "bb", "ccc"])

    scorer = Scorer.from_f_logpmf_to_batch_num(f)
    samples = scorer(d)
    assert [s.item for s in samples] == ["bb", "ccc", "a"]


def test_scorer_from_f_str_to_item() -> None:
    def f(s: str) -> ScoredItem[str]:
        time.sleep(1e-2)
        return ScoredItem(item=s + " ", score=len(s) + 1)

    d = LogPMF.from_samples(["a", "bb", "ccc"])

    scorer = Scorer.from_f_str_to_item(f)
    t1 = time.time()
    samples = scorer(d)
    t1 = time.time() - t1
    assert [s.score for s in samples] == [4, 3, 2]
    assert [s.item for s in samples] == ["ccc ", "bb ", "a "]

    scorer = Scorer.from_f_str_to_item(f, parallelize=True)
    t2 = time.time()
    samples = scorer(d)
    t2 = time.time() - t2
    assert [s.score for s in samples] == [4, 3, 2]
    assert [s.item for s in samples] == ["ccc ", "bb ", "a "]

    max_time = 3e-2
    assert t2 < max_time <= t1


def test_scorer_from_f_batch_str_to_batch_item() -> None:
    def f(ss: Sequence[str]) -> list[ScoredItem[str]]:
        return [ScoredItem(item=s + " ", score=len(s) + 1) for s in ss]

    d = LogPMF.from_samples(["a", "bb", "ccc"])

    scorer = Scorer.from_f_batch_str_to_batch_item(f)
    samples = scorer(d)
    assert [s.score for s in samples] == [4, 3, 2]
    assert [s.item for s in samples] == ["ccc ", "bb ", "a "]


def test_scorer_from_f_logpmf_to_batch_item() -> None:
    def f(d: LogPMF[str]) -> list[ScoredItem[str]]:
        return [
            ScoredItem(item=item + " ", score=float(jnp.exp(logp) * (len(item) + 1)))
            for logp, item in d
        ]

    d = LogPMF.from_samples(["a", "bb", "bb", "bb", "ccc"])

    scorer = Scorer.from_f_logpmf_to_batch_item(f)
    samples = scorer(d)
    assert [s.item for s in samples] == ["bb ", "ccc ", "a "]
