"""
Scorers are objects that take an instance of `decoding.pmf.CategoricalLogPMF`
and return a list of `decoding.pmf.Sample` instances that are sorted by their
utility values. The utility values are computed by a function that is passed
to the constructor of the `Scorer` object.

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

from decoding.pmf import CategoricalLogPMF, Sample, make_samples, sort_samples
from decoding.types import NUM


@dataclass(frozen=True, kw_only=True)
class Scorer:
    """
    The `Scorer` class wraps and coordinates user-supplied scoring functions.
    """

    _f: Callable[[CategoricalLogPMF[str]], list[Sample[str]]]

    def __call__(self, d: CategoricalLogPMF[str]) -> list[Sample[str]]:
        """
        `__call__` is an alias for `score`.
        """
        return self.score(d)

    def score(self, d: CategoricalLogPMF[str]) -> list[Sample[str]]:
        """
        Process a `decoding.pmf.CategoricalLogPMF` instance and returns a list
        of `decoding.pmf.Sample` instances that are sorted by their utility values.

        Args:
            d: A `decoding.pmf.CategoricalLogPMF` instance.

        Returns:
            A list of `decoding.pmf.Sample` instances that are sorted
            by their utility values.

        Example:
            ```python
            from decoding.pmf import CategoricalLogPMF
            from decoding.scorers import Scorer

            scorer = Scorer.from_f_str_to_num(lambda x: len(x))
            d = CategoricalLogPMF.from_samples(["a", "bb", "ccc"])
            samples = scorer(d)
            assert samples[0].item == "ccc"
            assert samples[0].utility == 3
            ```

        """
        return sort_samples(self._f(d))

    @classmethod
    def from_f_str_to_num(
        cls, f: Callable[[str], NUM], *, parallelize: bool = False
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a string to a
        number. The `Scorer` object will then score a
        `decoding.pmf.CategoricalLogPMF` instance by applying this function to
        each of its categories.

        Args:
            f: A function that maps a string to a number.
            parallelize: A boolean indicating whether to parallelize
                the scoring process.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            from decoding.pmf import CategoricalLogPMF
            from decoding.scorers import Scorer

            scorer = Scorer.from_f_str_to_num(lambda x: len(x), parallelize=True)
            d = CategoricalLogPMF.from_samples(["a", "bb", "ccc"])
            samples = scorer(d)
            assert samples[-1].item == "a"
            assert samples[-1].utility == 1
            ```

        """

        def _f(d: CategoricalLogPMF[str]) -> list[Sample[str]]:
            if parallelize:
                with ThreadPoolExecutor() as e:
                    utilities = list(e.map(f, d.cats))
            else:
                utilities = list(map(f, d.cats))
            return make_samples(d.cats, utilities)

        return cls(_f=_f)

    @classmethod
    def from_f_batch_str_to_batch_num(
        cls, f: Callable[[Sequence[str]], Sequence[NUM]]
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a sequence of
        strings to a sequence of numbers. The `Scorer` object will then score
        a `decoding.pmf.CategoricalLogPMF` instance by applying this function
        to its categories.

        Args:
            f: A function that maps a sequence of strings to a sequence of numbers.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            from decoding.pmf import CategoricalLogPMF
            from decoding.scorers import Scorer

            scorer = Scorer.from_f_batch_str_to_batch_num(lambda x: [len(s) for s in x])
            d = CategoricalLogPMF.from_samples(["a", "bb", "ccc"])
            samples = scorer(d)
            assert samples[0].item == "ccc"
            assert samples[0].utility == 3
            ```

        """

        def _f(d: CategoricalLogPMF[str]) -> list[Sample[str]]:
            utilities = f(d.cats)
            return make_samples(d.cats, utilities)

        return cls(_f=_f)

    @classmethod
    def from_f_catlogpmf_to_batch_num(
        cls, f: Callable[[CategoricalLogPMF[str]], Sequence[NUM]]
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a
        `decoding.pmf.CategoricalLogPMF` instance to a sequence of numbers.
        The `Scorer` object will then score the
        `decoding.pmf.CategoricalLogPMF` instance directly.

        Args:
            f: A function that maps a `decoding.pmf.CategoricalLogPMF`
                instance to a sequence of numbers.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            import jax.numpy as jnp
            from decoding.pmf import CategoricalLogPMF
            from decoding.scorers import Scorer

            f = lambda d: [jnp.exp(logp) * len(cat) for logp, cat in d]
            scorer = Scorer.from_f_catlogpmf_to_batch_num(f)
            d = CategoricalLogPMF.from_samples(["a", "bb", "bb", "ccc"])
            samples = scorer(d)
            assert samples[0].item == "bb"
            assert samples[0].utility == 1.0
            assert samples[1].item == "ccc"
            assert samples[1].utility == 0.75
            assert samples[2].item == "a"
            assert samples[2].utility == 0.25
            ```

        """

        def _f(d: CategoricalLogPMF[str]) -> list[Sample[str]]:
            utilities = f(d)
            return make_samples(d.cats, utilities)

        return cls(_f=_f)

    @classmethod
    def from_f_str_to_sample(
        cls, f: Callable[[str], Sample[str]], *, parallelize: bool = False
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a string to a
        `decoding.pmf.Sample` instance. The `Scorer` object will then score a
        `decoding.pmf.CategoricalLogPMF` instance by applying this function to
        each of its categories. This allows us to update not only the utility
        values but also the items themselves.

        Args:
            f: A function that maps a string to a `decoding.pmf.Sample` instance.
            parallelize: A boolean indicating whether to parallelize
                the scoring process.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            from decoding.pmf import CategoricalLogPMF, Sample
            from decoding.scorers import Scorer

            def f(x):
                if x.endswith("."):
                    return Sample(item=x[:-1], utility=len(x)-1)
                return Sample(item=x, utility=len(x))

            scorer = Scorer.from_f_str_to_sample(f, parallelize=True)
            d = CategoricalLogPMF.from_samples(["a", "bb.", "ccc"])
            samples = scorer(d)
            assert samples[0].item == "ccc"
            assert samples[0].utility == 3
            assert samples[1].item == "bb"
            assert samples[1].utility == 2
            ```

        """

        def _f(d: CategoricalLogPMF[str]) -> list[Sample[str]]:
            if parallelize:
                with ThreadPoolExecutor() as e:
                    return list(e.map(f, d.cats))
            else:
                return list(map(f, d.cats))

        return cls(_f=_f)

    @classmethod
    def from_f_batch_str_to_batch_sample(
        cls, f: Callable[[Sequence[str]], Sequence[Sample[str]]]
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a sequence of strings
        to a sequence of `decoding.pmf.Sample` instances. The `Scorer` object will
        then score a `decoding.pmf.CategoricalLogPMF` instance by applying this function
        to its categories. This allows us to update not only the utility values but
        also the items themselves.

        Args:
            f: A function that maps a sequence of strings to a sequence of
                `decoding.pmf.Sample` instances.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            from decoding.pmf import CategoricalLogPMF, Sample
            from decoding.scorers import Scorer

            f = lambda xs: [Sample(item=x[1:], utility=len(x[1:])) for x in xs]
            scorer = Scorer.from_f_batch_str_to_batch_sample(f)
            d = CategoricalLogPMF.from_samples(["_a", "_bb", "_ccc"])
            samples = scorer(d)
            assert samples[0].item == "ccc"
            assert samples[0].utility == 3
            ```

        """

        def _f(d: CategoricalLogPMF[str]) -> list[Sample[str]]:
            return list(f(d.cats))

        return cls(_f=_f)

    @classmethod
    def from_f_catlogpmf_to_batch_sample(
        cls, f: Callable[[CategoricalLogPMF[str]], Sequence[Sample[str]]]
    ) -> "Scorer":
        """
        Construct a `Scorer` object from a function that maps a
        `decoding.pmf.CategoricalLogPMF` instance to a sequence of
        `decoding.pmf.Sample` instances. This type signature actually
        matches much of the `decoding.estimators` module, so this constructor
        is particularly useful for building `Scorer` instances based on
        `decoding.estimators.MBR`, etc.

        Args:
            f: A function that maps a `decoding.pmf.CategoricalLogPMF`
                instance to a sequence of `decoding.pmf.Sample` instances.

        Returns:
            A `Scorer` object.

        Example:
            ```python
            import jax.numpy as jnp
            from decoding.estimators import MBR
            from decoding.pmf import CategoricalLogPMF
            from decoding.scorers import Scorer

            f = lambda d: MBR(d, utility=lambda x1, x2: x1 < x2)
            scorer = Scorer.from_f_catlogpmf_to_batch_sample(f)
            d = CategoricalLogPMF.from_samples(["aa", "bb", "cc"])
            samples = scorer(d)
            assert samples[0].item == "aa"
            assert jnp.isclose(samples[0].utility, 2/3)
            assert samples[1].item == "bb"
            assert jnp.isclose(samples[1].utility, 1/3)
            ```

        """

        def _f(d: CategoricalLogPMF[str]) -> list[Sample[str]]:
            return list(f(d))

        return cls(_f=_f)
