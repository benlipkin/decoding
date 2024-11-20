"""
Functions for sampling from instances of `decoding.pmf.LogPMF`.
"""

from collections.abc import Sequence

import jax.numpy as jnp
import jax.random as jr

from decoding.pmf import LogPMF
from decoding.types import FVX, IVY, KEY, T
from decoding.utils import getkey


def greedy(d: LogPMF[T], *, n: int = 1) -> T | list[T]:
    """
    Greedy decoding: return the most probable category.

    Args:
        d: A categorical distribution.
        n: The number of samples to return.

    Returns:
        The most probable category, or a list of n copies of it.

    Raises:
        ValueError: If n < 1.

    Example:
        ```python
        from decoding.pmf import LogPMF
        from decoding.samplers import greedy

        d = LogPMF.from_samples(["a", "b", "b"])
        assert greedy(d) == "b"
        ```

    """
    _validate_n(n)
    c = list(d.items)[jnp.argmax(d.logp).item()]
    return c if n == 1 else [c] * n


def random(d: LogPMF[T], *, n: int = 1, key: KEY | None = None) -> T | list[T]:
    """
    Random sampling: return a category sampled uniformly at random.

    Args:
        d: A categorical distribution.
        n: The number of samples to return.
        key: A PRNG key.

    Returns:
        A category sampled uniformly at random, or a list of n samples.

    Raises:
        ValueError: If n < 1.

    Example:
        ```python
        from decoding.pmf import LogPMF
        from decoding.samplers import random

        d = LogPMF.from_samples(["a", "b", "b"])
        assert random(d) in ["a", "b"]
        ```

    """
    _validate_n(n)
    opts = jnp.arange(len(d.items))
    return _sample(d.logp, d.items, opts, n, key)


def topk(d: LogPMF[T], *, k: int, n: int = 1, key: KEY | None = None) -> T | list[T]:
    """
    Top-k sampling: return a category from the top-k most probable categories.

    Args:
        d: A categorical distribution.
        k: The number of categories to consider.
        n: The number of samples to return.
        key: A PRNG key.

    Returns:
        A category from the top-k most probable categories, or a list of n samples.

    Raises:
        ValueError: If k < 1 or n < 1.

    Example:
        ```python
        from decoding.pmf import LogPMF
        from decoding.samplers import topk

        d = LogPMF.from_samples(["a", "b", "b"])
        assert topk(d, k=1) == "b"
        ```

    """
    _validate_n(n)
    if k <= 1:
        return greedy(d, n=n)
    if k >= len(d.items):
        return random(d, n=n)
    opts = jnp.argsort(d.logp, descending=True)[:k]
    return _sample(d.logp, d.items, opts, n, key)


def topp(d: LogPMF[T], *, p: float, n: int = 1, key: KEY | None = None) -> T | list[T]:
    """
    Top-p sampling: return a category from the top-p most probable categories.

    Args:
        d: A categorical distribution.
        p: The cumulative probability threshold.
        n: The number of samples to return.
        key: A PRNG key.

    Returns:
        A category from the top-p most probable categories, or a list of n samples.

    Raises:
        ValueError: If n < 1 or not 0 <= p <= 1.

    Example:
        ```python
        from decoding.pmf import LogPMF
        from decoding.samplers import topp

        d = LogPMF.from_samples(["a", "b", "b"])
        assert topp(d, p=0.5) == "b"
        ```

    """
    _validate_p(p)
    _validate_n(n)
    sort = jnp.argsort(d.logp, descending=True)
    opts = sort[jnp.cumsum(jnp.exp(d.logp[sort])) <= p]
    if (len(opts) == 0) | (p == 0):
        return greedy(d, n=n)
    if (len(opts) == len(d.items)) | (p == 1):
        return random(d, n=n)
    return _sample(d.logp, d.items, opts, n, key)


def minp(d: LogPMF[T], *, p: float, n: int = 1, key: KEY | None = None) -> T | list[T]:
    """
    Min-p sampling: return a category from the categories with probability at least p.

    Args:
        d: A categorical distribution.
        p: The probability threshold.
        n: The number of samples to return.
        key: A PRNG key.

    Returns:
        A category with probability at least p, or a list of n samples.

    Raises:
        ValueError: If n < 1 or not 0 <= p <= 1.

    Example:
        ```python
        from decoding.pmf import LogPMF
        from decoding.samplers import minp

        d = LogPMF.from_samples(["a", "b", "b"])
        assert minp(d, p=0.5) == "b"
        ```

    """
    _validate_p(p)
    _validate_n(n)
    if jnp.max(d.logp) < jnp.log(p):
        return greedy(d, n=n)
    if jnp.min(d.logp) > jnp.log(p):
        return random(d, n=n)
    opts = jnp.arange(len(d.items))[d.logp >= jnp.log(p)]
    return _sample(d.logp, d.items, opts, n, key)


def _sample(
    logp: FVX, items: Sequence[T], opts: IVY, n: int, key: KEY | None
) -> T | list[T]:
    if key is None:
        key = getkey()
    pmf = jnp.exp(logp[opts])
    pmf /= jnp.sum(pmf)
    indices = jr.choice(key, a=opts, p=pmf, shape=(n,), replace=True)
    return items[indices.item()] if n == 1 else [items[idx] for idx in indices.tolist()]


def _validate_n(n: int) -> None:
    if n < 1:
        msg = "n must be at least 1"
        raise ValueError(msg)


def _validate_p(p: float) -> None:
    if not 0 <= p <= 1:
        msg = "p must be in [0, 1]"
        raise ValueError(msg)
