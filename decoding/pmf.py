"""
Data structures and functions for working with probability mass functions.

This module provides a `CategoricalLogPMF` dataclass for working with categorical
probability mass functions (PMFs) in log-space. It also provides functions for
calculating various information-theoretic quantities, such as `surprise`, `entropy`,
`kl_divergence`, `cross_entropy`, etc.

The module also provides a `Sample` dataclass, instances of which are used to
store an `item` and its `utility` (e.g., a score, probability, or other measure).
There are also functions for creating and sorting lists of `Sample` instances.
"""

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeGuard

import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp

from decoding.types import FS, FVX, IVX, NUM, T
from decoding.utils import logsoftmax


@dataclass(frozen=True, kw_only=True)
class Sample(Generic[T]):
    """
    Dataclass for storing an item and its utility.

    Attributes:
        item: The item to be stored.
        utility: The utility of the item

    Example:
        ```python
        from decoding.pmf import Sample

        s = Sample(item="a", utility=0.5)
        assert s.item == "a"
        assert s.utility == 0.5
        ```

    """

    item: T
    utility: NUM


def sort_samples(samples: Iterable[Sample[T]]) -> list[Sample[T]]:
    """
    Sort a list of `Sample` instances by utility in descending order.

    Args:
        samples: An iterable of `Sample` instances.

    Returns:
        A list of `Sample` instances sorted by utility in descending order.

    Example:
        ```python
        from decoding.pmf import Sample, sort_samples

        samples = [
            Sample(item="a", utility=0.5),
            Sample(item="b", utility=0.3),
            Sample(item="c", utility=0.7),
        ]
        sorted_samples = sort_samples(samples)
        assert sorted_samples[0] == Sample(item="c", utility=0.7)
        ```

    """
    return sorted(samples, key=lambda x: float(x.utility), reverse=True)


def make_samples(items: Sequence[T], utilities: Sequence[NUM]) -> list[Sample[T]]:
    """
    Create a list of `Sample` instances from a list of items and utilities.

    Args:
        items: A sequence of items to be stored.
        utilities: A sequence of utilities for the items.

    Returns:
        A list of `Sample` instances.

    Example:
        ```python
        from decoding.pmf import make_samples

        items = ["a", "b", "c"]
        utilities = [0.5, 0.3, 0.7]
        samples = make_samples(items, utilities)
        assert samples[0] == Sample(item="a", utility=0.5)
        ```

    """
    return [Sample(item=i, utility=u) for i, u in zip(items, utilities, strict=True)]


@dataclass(frozen=True, kw_only=True)
class CategoricalLogPMF(Generic[T]):
    """
    Dataclass for working with categorical probability mass functions (PMFs)
    in log-space.

    Attributes:
        logp: The log-probabilities of the categories.
        cats: The categories themselves.

    Example:
        ```python
        import jax.numpy as jnp

        from decoding.pmf import CategoricalLogPMF

        logp = jnp.log(jnp.asarray([0.5, 0.3, 0.2]))
        cats = ["a", "b", "c"]
        d = CategoricalLogPMF(logp=logp, cats=cats)
        assert d.logp[0] == jnp.log(0.5)
        assert d.cats[0] == "a"
        ```

    """

    logp: FVX
    cats: Sequence[T]

    def __post_init__(self) -> None:
        """
        Validate the log-probabilities and categories.

        Raises:
            ValueError: If the log-probabilities array is not 1D,
                the log-probabilities and categories do not match in length,
                or the log-probabilities are not a proper distribution, i.e.,
                they do not sum to 1.

        """
        if len(self.logp.shape) != 1:
            msg = "LogProbs must be 1D"
            raise ValueError(msg)
        if len(self.logp) != len(self.cats):
            msg = "LogProbs and Categories must match length"
            raise ValueError(msg)
        if not jnp.isclose(jnn.logsumexp(self.logp), 0.0, atol=1e-3):
            msg = "LogProbs must be proper distribution"
            raise ValueError(msg)

    def __iter__(self) -> Iterator[tuple[FS, T]]:
        """
        Iterate over the log-probabilities and categories.

        Returns:
            An iterator over tuples of log-probabilities and categories.

        Example:
            ```python
            import jax.numpy as jnp

            from decoding.pmf import CategoricalLogPMF

            logp = jnp.log(jnp.asarray([0.5, 0.3, 0.2]))
            cats = ["a", "b", "c"]
            d = CategoricalLogPMF(logp=logp, cats=cats)
            for lp, cat in d:
                assert lp == jnp.log(0.5)
                assert cat == "a"
                break
            ```

        """
        return zip(self.logp, self.cats, strict=True)

    @classmethod
    def from_logits(
        cls, *, logits: FVX, cats: Sequence[T], temp: float = 1.0
    ) -> "CategoricalLogPMF[T]":
        """
        Create a `CategoricalLogPMF` instance from logits.

        Args:
            logits: The logits of the categories.
            cats: The categories themselves.
            temp: The temperature for the softmax function.

        Returns:
            A `CategoricalLogPMF` instance.

        Example:
            ```python
            import jax.numpy as jnp
            import jax.nn as jnn

            from decoding.pmf import CategoricalLogPMF

            logits = jnp.asarray([1.0, 2.0, 3.0])
            cats = ["a", "b", "c"]
            d = CategoricalLogPMF.from_logits(logits=logits, cats=cats)
            assert jnn.logsumexp(d.logp) == 0.0
            ```

        """
        return cls(logp=logsoftmax(logits, t=temp), cats=cats)

    @classmethod
    def from_samples(
        cls, samples: Sequence[T] | Sequence[Sample[T]]
    ) -> "CategoricalLogPMF[T]":
        """
        Create a `CategoricalLogPMF` instance from a list of items
        or a list of `Sample` instances.

        Args:
            samples: A sequence of items or `Sample` instances.

        Returns:
            A `CategoricalLogPMF` instance.

        Example:
            ```python
            from decoding.pmf import CategoricalLogPMF, Sample

            samples = ["a", "b", "a", "c"]
            d = CategoricalLogPMF.from_samples(samples)
            assert d.logp[0] == jnp.log(0.5)
            assert d.cats[0] == "a"
            assert d.logp[1] == jnp.log(0.25)
            assert d.cats[1] == "b"
            ```

        """
        items = _prepare_items(samples)
        ht = Counter(items)
        cats = list(ht.keys())
        counts = jnp.asarray(list(ht.values()))
        mle = jnp.log(counts) - jnp.log(jnp.sum(counts))
        return cls(logp=mle, cats=cats)


def surprise(d: CategoricalLogPMF[T], cat: T) -> FS:
    """
    Calculate the surprise of a category in a categorical distribution.

    Args:
        d: The categorical distribution.
        cat: The category of interest.

    Returns:
        The surprise of the category.

    Raises:
        ValueError: If the category is not in the distribution.

    Example:
        ```python
        import jax.numpy as jnp

        from decoding.pmf import CategoricalLogPMF, surprise

        logp = jnp.log(jnp.asarray([0.5, 0.5]))
        cats = ["a", "b"]
        d = CategoricalLogPMF(logp=logp, cats=cats)
        assert surprise(d, "a") / jnp.log(2) == 1.0 # 1 bit
        ```

    """
    indices = jnp.asarray([i for i, c in enumerate(d.cats) if c == cat]).astype(int)
    _validate_indices(indices)
    return -jnn.logsumexp(d.logp[indices])


def entropy(d: CategoricalLogPMF[T]) -> FS:
    """
    Calculate the entropy of a categorical distribution.

    Args:
        d: The categorical distribution.

    Returns:
        The entropy of the distribution.

    Example:
        ```python
        import jax.numpy as jnp

        from decoding.pmf import CategoricalLogPMF, entropy

        logp = jnp.log(jnp.asarray([0.5, 0.5]))
        cats = ["a", "b"]
        d = CategoricalLogPMF(logp=logp, cats=cats)
        assert entropy(d) / jnp.log(2) == 1.0 # 1 bit
        ```

    """
    vals = d.logp * jnp.exp(d.logp)
    nanmask = ~jnp.isnan(vals)
    return -jnp.sum(vals[nanmask])


def kl_divergence(d_p: CategoricalLogPMF[T], d_q: CategoricalLogPMF[T]) -> FS:
    """
    Calculate the KL-divergence between two categorical distributions.

    Args:
        d_p: The first categorical distribution.
        d_q: The second categorical distribution.

    Returns:
        The KL-divergence between the distributions.

    Raises:
        ValueError: If the distributions do not have the same categories.

    Example:
        ```python
        import jax.numpy as jnp

        from decoding.pmf import CategoricalLogPMF, kl_divergence

        logp_p = jnp.log(jnp.asarray([1.0, 0.0]))
        logp_q = jnp.log(jnp.asarray([0.5, 0.5]))
        cats = ["a", "b"]
        d_p = CategoricalLogPMF(logp=logp_p, cats=cats)
        d_q = CategoricalLogPMF(logp=logp_q, cats=cats)
        assert kl_divergence(d_p, d_q) / jnp.log(2) == 1.0 # 1 bit
        ```

    """
    _validate_cats(d_p.cats, d_q.cats)
    return jnp.sum(jsp.special.kl_div(jnp.exp(d_p.logp), jnp.exp(d_q.logp)))


def cross_entropy(d_p: CategoricalLogPMF[T], d_q: CategoricalLogPMF[T]) -> FS:
    """
    Calculate the cross-entropy between two categorical distributions.

    Args:
        d_p: The first categorical distribution.
        d_q: The second categorical distribution.

    Returns:
        The cross-entropy between the distributions.

    Raises:
        ValueError: If the distributions do not have the same categories.

    Example:
        ```python
        import jax.numpy as jnp

        from decoding.pmf import CategoricalLogPMF, cross_entropy

        logp_p = jnp.log(jnp.asarray([1.0, 0.0]))
        logp_q = jnp.log(jnp.asarray([0.5, 0.5]))
        cats = ["a", "b"]
        d_p = CategoricalLogPMF(logp=logp_p, cats=cats)
        d_q = CategoricalLogPMF(logp=logp_q, cats=cats)
        assert cross_entropy(d_p, d_q) / jnp.log(2) == 1.0 # 1 bit
        ```

    """
    return entropy(d_p) + kl_divergence(d_p, d_q)


def js_divergence(d_p: CategoricalLogPMF[T], d_q: CategoricalLogPMF[T]) -> FS:
    """
    Calculate the Jensen-Shannon divergence between two categorical distributions.

    Args:
        d_p: The first categorical distribution.
        d_q: The second categorical distribution.

    Returns:
        The Jensen-Shannon divergence between the distributions.

    Raises:
        ValueError: If the distributions do not have the same categories.

    Example:
        ```python
        import jax.numpy as jnp

        from decoding.pmf import CategoricalLogPMF, js_divergence

        logp = jnp.log(jnp.asarray([0.5, 0.5]))
        cats = ["a", "b"]
        d_p = CategoricalLogPMF(logp=logp, cats=cats)
        d_q = CategoricalLogPMF(logp=logp, cats=cats)
        assert js_divergence(d_p, d_q) == 0.0
        ```

    """
    _validate_cats(d_p.cats, d_q.cats)
    avg = CategoricalLogPMF[T](
        logp=jnp.logaddexp(d_p.logp, d_q.logp) - jnp.log(2),
        cats=d_p.cats,
    )
    return (kl_divergence(d_p, avg) + kl_divergence(d_q, avg)) / 2


def js_distance(d_p: CategoricalLogPMF[T], d_q: CategoricalLogPMF[T]) -> FS:
    """
    Calculate the Jensen-Shannon distance between two categorical distributions.

    Args:
        d_p: The first categorical distribution.
        d_q: The second categorical distribution.

    Returns:
        The Jensen-Shannon distance between the distributions.

    Raises:
        ValueError: If the distributions do not have the same categories.

    Example:
        ```python
        import jax.numpy as jnp

        from decoding.pmf import CategoricalLogPMF, js_distance

        logp = jnp.log(jnp.asarray([0.5, 0.5]))
        cats = ["a", "b"]
        d_p = CategoricalLogPMF(logp=logp, cats=cats)
        d_q = CategoricalLogPMF(logp=logp, cats=cats)
        assert js_distance(d_p, d_q) == 0.0
        ```

    """
    return jnp.sqrt(js_divergence(d_p, d_q))


def _prepare_items(samples: Sequence[T] | Sequence[Sample[T]]) -> Sequence[T]:
    if _guard_sample_seq(samples):
        return [s.item for s in samples]
    if _guard_item_seq(samples):
        return samples
    msg = "Samples must be `Sequence[T]` or `Sequence[Sample[T]]` and nonempty"
    raise ValueError(msg)


def _guard_sample_seq(
    samples: Sequence[T] | Sequence[Sample[T]],
) -> TypeGuard[Sequence[Sample[T]]]:
    if len(samples) == 0:
        return False
    return isinstance(samples[0], Sample)


def _guard_item_seq(
    samples: Sequence[T] | Sequence[Sample[T]],
) -> TypeGuard[Sequence[T]]:
    return len(samples) > 0


def _validate_cats(pcats: Sequence[T], qcats: Sequence[T]) -> None:
    if pcats != qcats:
        msg = "Distributions must have the same categories"
        raise ValueError(msg)


def _validate_indices(indices: IVX) -> None:
    if len(indices) == 0:
        msg = "Category not in distribution"
        raise ValueError(msg)
