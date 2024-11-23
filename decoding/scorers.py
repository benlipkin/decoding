"""
Scorers are objects that take an instance of `decoding.pmf.LogPMF`
and return a list of `decoding.pmf.ScoredItem` instances that are sorted by their
scores. The scores are computed by a function that is passed to the constructor
of the `Scorer` object.

The `Scorer` class is a frozen dataclass that wraps this scoring function. The
class supports constructors that enable the preparation of this scoring function
from a variety of input types, depending on what is most convenient for the user.
For example, the user can choose whether to engage with or simply allow the class
to coordinate details like batching, weighting, and parallelization.

**NB**: The examples below are illustrative of the API, but are not particularly
meaningful or interesting. Check out the
[`TUTORIAL.md`](https://github.com/benlipkin/decoding/blob/main/TUTORIAL.md)
for more practical examples of scoring functions in action.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from decoding.pmf import LogPMF, ScoredItem, make_scored_items, sort_scored_items
from decoding.types import NUM


@dataclass(frozen=True, kw_only=True)
class Scorer:
    """
    The `Scorer` class wraps and coordinates user-supplied scoring functions.
    """

    _f: Callable[[LogPMF[str]], list[ScoredItem[str]]]

    def __call__(self, d: LogPMF[str]) -> list[ScoredItem[str]]:
        """
        `__call__` is an alias for `score`.
        """
        return self.score(d)

    def score(self, d: LogPMF[str]) -> list[ScoredItem[str]]:
        """
        Process a `decoding.pmf.LogPMF` instance and returns a list
        of `decoding.pmf.ScoredItem` instances that are sorted by their scores.

        Args:
            d: A `decoding.pmf.LogPMF` instance.

        Returns:
            A list of `decoding.pmf.ScoredItem` instances that are sorted
            by their scores.

        Example:
            ```python
            from decoding.pmf import LogPMF
            from decoding.scorers import Scorer

            scorer = Scorer.from_f_str_to_num(lambda x: len(x))
            d = LogPMF.from_samples(["a", "bb", "ccc"])
            samples = scorer(d)
            assert samples[0].item == "ccc"
            assert samples[0].score == 3
            ```

        """
        return sort_scored_items(self._f(d))

    @classmethod
    def from_f_str_to_num(
        cls, f: Callable[[str], NUM], *, parallelize: bool = False
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a string to a
        number. The `Scorer` object will then score a
        `decoding.pmf.LogPMF` instance by applying this function to
        each of its categories.

        Args:
            f: A function that maps a string to a number.
            parallelize: A boolean indicating whether to parallelize
                the scoring process.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            from decoding.pmf import LogPMF
            from decoding.scorers import Scorer

            scorer = Scorer.from_f_str_to_num(lambda x: len(x), parallelize=True)
            d = LogPMF.from_samples(["a", "bb", "ccc"])
            samples = scorer(d)
            assert samples[-1].item == "a"
            assert samples[-1].score == 1
            ```

        """

        def _f(d: LogPMF[str]) -> list[ScoredItem[str]]:
            if parallelize:
                with ThreadPoolExecutor() as e:
                    utilities = list(e.map(f, d.items))
            else:
                utilities = list(map(f, d.items))
            return make_scored_items(d.items, utilities)

        return cls(_f=_f)

    @classmethod
    def from_f_batch_str_to_batch_num(
        cls, f: Callable[[Sequence[str]], Sequence[NUM]]
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a sequence of
        strings to a sequence of numbers. The `Scorer` object will then score
        a `decoding.pmf.LogPMF` instance by applying this function
        to its categories.

        Args:
            f: A function that maps a sequence of strings to a sequence of numbers.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            from decoding.pmf import LogPMF
            from decoding.scorers import Scorer

            scorer = Scorer.from_f_batch_str_to_batch_num(lambda x: [len(s) for s in x])
            d = LogPMF.from_samples(["a", "bb", "ccc"])
            samples = scorer(d)
            assert samples[0].item == "ccc"
            assert samples[0].score == 3
            ```

        """

        def _f(d: LogPMF[str]) -> list[ScoredItem[str]]:
            utilities = f(d.items)
            return make_scored_items(d.items, utilities)

        return cls(_f=_f)

    @classmethod
    def from_f_logpmf_to_batch_num(
        cls, f: Callable[[LogPMF[str]], Sequence[NUM]]
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a
        `decoding.pmf.LogPMF` instance to a sequence of numbers.
        The `Scorer` object will then score the
        `decoding.pmf.LogPMF` instance directly.

        Args:
            f: A function that maps a `decoding.pmf.LogPMF`
                instance to a sequence of numbers.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            import jax.numpy as jnp
            from decoding.pmf import LogPMF
            from decoding.scorers import Scorer

            f = lambda d: [jnp.exp(logp) * len(item) for logp, item in d]
            scorer = Scorer.from_f_logpmf_to_batch_num(f)
            d = LogPMF.from_samples(["a", "bb", "bb", "ccc"])
            samples = scorer(d)
            assert samples[0].item == "bb"
            assert samples[0].score == 1.0
            assert samples[1].item == "ccc"
            assert samples[1].score == 0.75
            assert samples[2].item == "a"
            assert samples[2].score == 0.25
            ```

        """

        def _f(d: LogPMF[str]) -> list[ScoredItem[str]]:
            utilities = f(d)
            return make_scored_items(d.items, utilities)

        return cls(_f=_f)

    @classmethod
    def from_f_str_to_item(
        cls, f: Callable[[str], ScoredItem[str]], *, parallelize: bool = False
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a string to a
        `decoding.pmf.ScoredItem` instance. The `Scorer` object will then score a
        `decoding.pmf.LogPMF` instance by applying this function to
        each of its categories. This allows us to update not only the score
        values but also the items themselves.

        Args:
            f: A function that maps a string to a `decoding.pmf.ScoredItem` instance.
            parallelize: A boolean indicating whether to parallelize
                the scoring process.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            from decoding.pmf import LogPMF, ScoredItem
            from decoding.scorers import Scorer

            def f(x):
                if x.endswith("."):
                    return ScoredItem(item=x[:-1], score=len(x)-1)
                return ScoredItem(item=x, score=len(x))

            scorer = Scorer.from_f_str_to_item(f, parallelize=True)
            d = LogPMF.from_samples(["a", "bb.", "ccc"])
            samples = scorer(d)
            assert samples[0].item == "ccc"
            assert samples[0].score == 3
            assert samples[1].item == "bb"
            assert samples[1].score == 2
            ```

        """

        def _f(d: LogPMF[str]) -> list[ScoredItem[str]]:
            if parallelize:
                with ThreadPoolExecutor() as e:
                    return list(e.map(f, d.items))
            else:
                return list(map(f, d.items))

        return cls(_f=_f)

    @classmethod
    def from_f_batch_str_to_batch_item(
        cls, f: Callable[[Sequence[str]], Sequence[ScoredItem[str]]]
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a sequence of strings
        to a sequence of `decoding.pmf.ScoredItem` instances. The `Scorer` object will
        then score a `decoding.pmf.LogPMF` instance by applying this function
        to its categories. This allows us to update not only the score values but
        also the items themselves.

        Args:
            f: A function that maps a sequence of strings to a sequence of
                `decoding.pmf.ScoredItem` instances.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            from decoding.pmf import LogPMF, ScoredItem
            from decoding.scorers import Scorer

            f = lambda xs: [ScoredItem(item=x[1:], score=len(x[1:])) for x in xs]
            scorer = Scorer.from_f_batch_str_to_batch_item(f)
            d = LogPMF.from_samples(["_a", "_bb", "_ccc"])
            samples = scorer(d)
            assert samples[0].item == "ccc"
            assert samples[0].score == 3
            ```

        """

        def _f(d: LogPMF[str]) -> list[ScoredItem[str]]:
            return list(f(d.items))

        return cls(_f=_f)

    @classmethod
    def from_f_logpmf_to_batch_item(
        cls, f: Callable[[LogPMF[str]], Sequence[ScoredItem[str]]]
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a
        `decoding.pmf.LogPMF` instance to a sequence of
        `decoding.pmf.ScoredItem` instances. This type signature actually
        matches much of the `decoding.estimators` module, so this constructor
        is particularly useful for building `Scorer` instances based on
        `decoding.estimators.MBR`, etc.

        Args:
            f: A function that maps a `decoding.pmf.LogPMF`
                instance to a sequence of `decoding.pmf.ScoredItem` instances.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            import jax.numpy as jnp
            from decoding.estimators import MBR
            from decoding.pmf import LogPMF
            from decoding.scorers import Scorer

            f = lambda d: MBR(d, utility=lambda x1, x2: x1 < x2)
            scorer = Scorer.from_f_logpmf_to_batch_item(f)
            d = LogPMF.from_samples(["aa", "bb", "cc"])
            samples = scorer(d)
            assert samples[0].item == "aa"
            assert jnp.isclose(samples[0].score, 2/3)
            assert samples[1].item == "bb"
            assert jnp.isclose(samples[1].score, 1/3)
            ```

        """

        def _f(d: LogPMF[str]) -> list[ScoredItem[str]]:
            return list(f(d))

        return cls(_f=_f)
