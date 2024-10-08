import time
from collections.abc import Sequence

import jax.numpy as jnp

from decoding.pmf import CategoricalLogPMF, Sample
from decoding.scorers import Scorer


def test_scorer_from_f_str_to_num() -> None:
    def f(s: str) -> int:
        time.sleep(1e-2)
        return len(s)

    d = CategoricalLogPMF.from_samples(["a", "bb", "ccc"])

    scorer = Scorer.from_f_str_to_num(f)
    t1 = time.time()
    samples = scorer(d)
    t1 = time.time() - t1
    assert [s.utility for s in samples] == [3, 2, 1]

    scorer = Scorer.from_f_str_to_num(f, parallelize=True)
    t2 = time.time()
    samples = scorer(d)
    t2 = time.time() - t2
    assert [s.utility for s in samples] == [3, 2, 1]

    max_time = 3e-2
    assert t2 < max_time <= t1


def test_scorer_from_f_batch_str_to_batch_num() -> None:
    def f(ss: Sequence[str]) -> list[int]:
        return [len(s) for s in ss]

    d = CategoricalLogPMF.from_samples(["a", "bb", "ccc"])

    scorer = Scorer.from_f_batch_str_to_batch_num(f)
    samples = scorer(d)
    assert [s.utility for s in samples] == [3, 2, 1]


def test_scorer_from_f_catlogpmf_to_batch_num() -> None:
    def f(d: CategoricalLogPMF[str]) -> list[float]:
        return [float(jnp.exp(logp) * len(cat)) for logp, cat in d]

    d = CategoricalLogPMF.from_samples(["a", "bb", "bb", "bb", "ccc"])

    scorer = Scorer.from_f_catlogpmf_to_batch_num(f)
    samples = scorer(d)
    assert [s.item for s in samples] == ["bb", "ccc", "a"]


def test_scorer_from_f_str_to_sample() -> None:
    def f(s: str) -> Sample[str]:
        time.sleep(1e-2)
        return Sample(item=s + " ", utility=len(s) + 1)

    d = CategoricalLogPMF.from_samples(["a", "bb", "ccc"])

    scorer = Scorer.from_f_str_to_sample(f)
    t1 = time.time()
    samples = scorer(d)
    t1 = time.time() - t1
    assert [s.utility for s in samples] == [4, 3, 2]
    assert [s.item for s in samples] == ["ccc ", "bb ", "a "]

    scorer = Scorer.from_f_str_to_sample(f, parallelize=True)
    t2 = time.time()
    samples = scorer(d)
    t2 = time.time() - t2
    assert [s.utility for s in samples] == [4, 3, 2]
    assert [s.item for s in samples] == ["ccc ", "bb ", "a "]

    max_time = 3e-2
    assert t2 < max_time <= t1


def test_scorer_from_f_batch_str_to_batch_sample() -> None:
    def f(ss: Sequence[str]) -> list[Sample[str]]:
        return [Sample(item=s + " ", utility=len(s) + 1) for s in ss]

    d = CategoricalLogPMF.from_samples(["a", "bb", "ccc"])

    scorer = Scorer.from_f_batch_str_to_batch_sample(f)
    samples = scorer(d)
    assert [s.utility for s in samples] == [4, 3, 2]
    assert [s.item for s in samples] == ["ccc ", "bb ", "a "]


def test_scorer_from_f_catlogpmf_to_batch_sample() -> None:
    def f(d: CategoricalLogPMF[str]) -> list[Sample[str]]:
        return [
            Sample(item=cat + " ", utility=float(jnp.exp(logp) * (len(cat) + 1)))
            for logp, cat in d
        ]

    d = CategoricalLogPMF.from_samples(["a", "bb", "bb", "bb", "ccc"])

    scorer = Scorer.from_f_catlogpmf_to_batch_sample(f)
    samples = scorer(d)
    assert [s.item for s in samples] == ["bb ", "ccc ", "a "]
