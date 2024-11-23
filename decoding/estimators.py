"""
Methods for calculating point estimates from distributions.

Estimators in this module operate over an instance of `decoding.pmf.LogPMF`
and return a list of `decoding.pmf.ScoredItem` instances sorted by their expected
utility. Each `decoding.pmf.ScoredItem` instance contains an `item` and `score` field.
More about these data structures can be found in the `decoding.pmf` module.

The estimators in this module reflect variants of the Minimum Bayes Risk (MBR). The MBR
is a decision-theoretic approach to point estimation that minimizes the expected loss
of a decision rule. This module provides efficient implementations of MBR that account
for the properties of arbitrary user-provided utility functions.

The module also provides a `MAP` estimator, which is a special case of `MBR` where the
utility function is the identity function, and a `SelfConsistency` estimator, which
applies a post-processing and filtering step before aggregating the resulting samples
via a majority voting procedure.
"""

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import jax.numpy as jnp

from decoding.pmf import LogPMF, ScoredItem, make_scored_items, sort_scored_items
from decoding.types import FS, NUM, T_, T


def MBR(
    d: LogPMF[T], *, utility: Callable[[T, T], NUM], parallelize: bool = False
) -> list[ScoredItem[T]]:
    """
    Calculate the Minimum Bayes Risk (MBR) estimator for a given distribution
    and arbitrary user-provided utility function.

    Args:
        d: The distribution to estimate the MBR from.
        utility: The utility function to use for the MBR calculation.
        parallelize: Whether to parallelize the utility calculation.

    Returns:
        A sorted list of `decoding.pmf.ScoredItem` instances by their expected utility.

    Example:
        ```python
        from decoding.estimators import MBR
        from decoding.pmf import LogPMF

        d = LogPMF.from_samples(["a","b","c"])
        samples = MBR(d, utility=lambda c1, c2: c1 > c2)
        assert samples[0].item == "c"
        ```

    """

    def _risk(c1: T) -> FS:
        return -jnp.sum(jnp.asarray([utility(c1, c2) for _, c2 in d])).astype(float)

    return _MBR(d, _risk, parallelize=parallelize)


def commutativeMBR(
    d: LogPMF[T], *, utility: Callable[[T, T], NUM], parallelize: bool = False
) -> list[ScoredItem[T]]:
    """
    Variant of `MBR` for commutative utility functions.
    By exploiting the commutative property of the utility function, this
    estimator reduces the number of utility calculations by half.

    Args:
        d: The distribution to estimate the MBR from.
        utility: The utility function to use for the MBR calculation.
        parallelize: Whether to parallelize the utility calculation.

    Returns:
        A sorted list of `decoding.pmf.ScoredItem` instances by their expected utility.

    Example:
        ```python
        from decoding.estimators import commutativeMBR
        from decoding.pmf import LogPMF

        d = LogPMF.from_samples(["a","b","b"])
        samples = commutativeMBR(d, utility=lambda c1, c2: c1 == c2)
        assert samples[0].item == "b"
        ```

    """

    @cache
    def _utility(cs: frozenset[T]) -> NUM:
        return utility(*cs, *cs) if len(cs) == 1 else utility(*cs)

    def _risk(c1: T) -> FS:
        return -jnp.sum(
            jnp.asarray([_utility(frozenset({c1, c2})) for _, c2 in d])
        ).astype(float)

    return _MBR(d, _risk, parallelize=parallelize)


def linearMBR(
    d: LogPMF[T], *, utility: Callable[[T], NUM], parallelize: bool = False
) -> list[ScoredItem[T]]:
    """
    Variant of `MBR` for cases that can be executed in linear time.
    By exploiting utility functions that operate only on individual elements,
    this estimator reduces the complexity of calculation from O(n^2) to O(n).

    Args:
        d: The distribution to estimate the MBR from.
        utility: The utility function to use for the MBR calculation.
        parallelize: Whether to parallelize the utility calculation.

    Returns:
        A sorted list of `decoding.pmf.ScoredItem` instances by their expected utility.

    Example:
        ```python
        from decoding.estimators import linearMBR
        from decoding.pmf import LogPMF

        d = LogPMF.from_samples(["a","bb","ccc"])
        samples = linearMBR(d, utility=lambda c: len(c))
        assert samples[0].item == "ccc"
        ```

    """

    def _risk(c1: T) -> FS:
        return -jnp.asarray(utility(c1)).astype(float)

    return _MBR(d, _risk, parallelize=parallelize)


def MAP(d: LogPMF[T], *, parallelize: bool = False) -> list[ScoredItem[T]]:
    """
    Calculate the Maximum A Posteriori (MAP) estimator for a given distribution.

    Args:
        d: The distribution to estimate the MAP from.
        parallelize: Whether to parallelize the utility calculation.

    Returns:
        A sorted list of `decoding.pmf.ScoredItem` instances by their expected utility.

    Example:
        ```python
        from decoding.estimators import MAP
        from decoding.pmf import LogPMF

        d = LogPMF.from_samples(["a","b","b"])
        samples = MAP(d)
        assert samples[0].item == "b"
        ```

    """

    def _utility(_: T) -> float:
        return 1.0

    return linearMBR(d, utility=_utility, parallelize=parallelize)


def SelfConsistency(
    d: LogPMF[T],
    *,
    postproc: Callable[[T], T_],
    filt: Callable[[T_], bool],
    parallelize: bool = False,
) -> list[ScoredItem[T_]]:
    """
    Calculate the Self-Consistency estimator for a given distribution, after applying
    a post-processing and filtering step.

    Args:
        d: The distribution to estimate the Self-Consistency from.
        postproc: The post-processing function to apply to the samples.
        filt: The filtering function to apply to the post-processed samples.
        parallelize: Whether to parallelize the utility calculation.

    Returns:
        A sorted list of `decoding.pmf.ScoredItem` instances by their expected utility.

    Example:
        ```python
        from decoding.estimators import SelfConsistency
        from decoding.pmf import LogPMF

        d = LogPMF.from_samples(["aa","ab","ba","bb","bc"])
        samples = SelfConsistency(d, postproc=lambda c: c[0], filt=lambda c: c != "b")
        assert samples[0].item == "a"
        ```

    """
    _postproc = cache(postproc)

    def _aggregate(
        samples: list[ScoredItem[T]],
        _postproc: Callable[[T], T_],
        filt: Callable[[T_], bool],
    ) -> list[ScoredItem[T_]]:
        ht = defaultdict(lambda: 0.0)
        for sample in samples:
            c = _postproc(sample.item)
            if filt(c):
                ht[c] += float(sample.score)
        return sort_scored_items([ScoredItem(item=c, score=u) for c, u in ht.items()])

    def _utility(c1: T, c2: T) -> int:
        return int(_postproc(c1) == _postproc(c2))

    samples = MBR(d, utility=_utility, parallelize=parallelize)
    return _aggregate(samples, _postproc, filt)


def _MBR(
    d: LogPMF[T], risk: Callable[[T], FS], *, parallelize: bool = False
) -> list[ScoredItem[T]]:
    def _calc_utility(logp: FS, c1: T) -> float:
        return -float(risk(c1) * jnp.exp(logp))

    if parallelize:
        with ThreadPoolExecutor() as e:
            utilities = list(e.map(_calc_utility, d.logp, d.items))
    else:
        utilities = list(map(_calc_utility, d.logp, d.items))
    return sort_scored_items(make_scored_items(d.items, utilities))
