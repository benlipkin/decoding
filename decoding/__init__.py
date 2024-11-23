"""
Composable inference algorithms with LLMs and programmable logic.

Check out the [`README`](https://github.com/benlipkin/decoding/blob/main/README.md)
and [`TUTORIAL`](https://github.com/benlipkin/decoding/blob/main/TUTORIAL.md)
on [GitHub](https://github.com/benlipkin/decoding/) for more information.

Core modules:
- `decoding.models`: Interface for working with Language Models (LMs), and
    leveraging them to generate text or score sequences.
- `decoding.scorers`: Interface for constructing custom scoring functions
    that can be used to rank and steer generation.
- `decoding.generators`: Interface for implementing custom generation algorithms
    that can be used to sample controlled text from LMs.

Supporting modules:
- `decoding.pmf`: Data structures for probability mass functions and other collections
    of measures as well as algorithms for calculating information-theoretic quantities.
- `decoding.samplers`: Methods for sampling from distributions.
- `decoding.estimators`: Decision rules for deriving point estimates from distributions.
    Supports a flexible Minimum Bayes Risk (MBR) interface that accepts arbitrary
    user-defined utility functions.
- `decoding.metrics`: Metrics that may be useful for constructing scoring functions.
- `decoding.utils`: Miscellaneous helper functions for the library.
"""

from decoding import (
    estimators,
    generators,
    metrics,
    models,
    pmf,
    samplers,
    scorers,
    utils,
)

__all__ = [
    "estimators",
    "generators",
    "metrics",
    "models",
    "pmf",
    "samplers",
    "scorers",
    "utils",
]
