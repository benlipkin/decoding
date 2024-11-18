# Decoding

[![Tests](https://github.com/benlipkin/decoding/actions/workflows/static.yml/badge.svg)](https://github.com/benlipkin/decoding/tree/main/tests)
[![Docs](https://github.com/benlipkin/decoding/actions/workflows/docs.yml/badge.svg)](https://benlipkin.github.io/decoding/decoding.html)

Composable inference algorithms with LLMs and programmable logic.

## Overview

`decoding` is a library for scaling the inference-time capabilities of LLMs, enabling users to easily solve difficult problems. The library is built around simple sampling and reranking patterns that accept arbitrary user-defined scoring functions. At its simplest, you write a function that takes in a string and returns a float, and we handle the rest. If you'd like to step things up, we provide simple patterns to enable the efficient construction of powerful algorithms like Backtracking Monte Carlo Tree Search variants.

### Why should I care?

Scaling inference thoughtfully is yielding breakthrough performance improvements in the world of LLMs. We are already seeing small models out-perform models >10x their size by leveraging basic sampling and search strategies, particularly in combination with custom verifiers, scoring functions, and process reward models. Check out this excellent recent [presentation](https://srush.github.io/awesome-o1/o1-tutorial.pdf) by Sasha Rush and Daniel Ritter for more background. `decoding` makes it effortless to explore this design space and allows researchers to quickly iterate on their ideas.

## Getting started

Install directly from [PyPi](https://pypi.org/project/decoding/)

```bash
python -m pip install decoding
```

See [Contributing](https://github.com/benlipkin/decoding#documentation) for how to build the dev and testing environment.

<u>NOTE:</u> Decoding depends on `vLLM`, which means this library can only be built on linux, and by default must be run on GPU. To run on CPU, see the [instructions from vLLM](https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html).

### Documentation

All modules, classes, and functions of our public interface are documented on our [website](https://benlipkin.github.io/decoding/). The docs should be the first stop for questions about the API. More realistic use cases and interesting patterns can be found in the tutorial.

### Tutorial

Several examples are provided to give you a taste and help you get started. Check out [`TUTORIAL.md`](https://github.com/benlipkin/decoding/blob/main/TUTORIAL.md) for a commented walk-through of a few sample design patterns, or go straight to the code and run it yourself in the [`examples/`](https://github.com/benlipkin/decoding/tree/main/examples) directory.

## The library's philosophy

__The most valuable resource is the researcher's time.__ The `decoding` library is designed from the ground up to easily, flexibly, and quickly support experimentation over new ideas, while coordinating all the engineering glue on the backend. We make a few design decisions that support this.

1. The library is built with an emphasis on pure functions and immutability. All classes are frozen dataclasses. All functions express typed, composable building blocks that flexibly wrap and coordinate any user-generated code. This enables users to get very creative with their design patterns without having to chase down complicated bugs.

2. When there is a decision point between squeezing a drop of performance or maintaining flexibility, we maintain flexibility. This may be controversial. When one decides a piece of code is meant to do only one thing, there are a myriad of optimizations that become available. But, if your use-case doesn't fall there, those optimizations aren't particularly useful for you who can't use the library. We optimize the library to support the broadest range of ideas a researcher can come up with, such as backtracking, resampling, or otherwise modifying in-progress generations.

3. We still keep things fast. We use libraries like `vLLM` under the hood to keep text generation fast, which is often the primary bottleneck. We also expose arguments for users to specify when parts of scorers or estimators can be run concurrently, and harness CPU-based parallelism for the heavier parts like executing LLM-generated code. More detailed profiling over common use cases is coming shortly, which will be used to drive future development.

Overall, these design decisions make `decoding` a great library for R&D. Researchers can flexibly explore the design space of relevant inference algoriths for a task, and when they've arrived at a solution that is optimal and they'd like to scale it, they can refactor the relevant bottlenecks and exploit the specific optimizations that are available for their individual case.

## What's next

There are a number of features coming soon to `decoding`. 

### Monte Carlo Tree Search

Currently in `decoding.experimental` we have initial support for a powerful `RolloutTreeSearch` algorithm. This wraps the `TreeSearch` interface, enabling rollouts within each sync phase and pushing scores back up the tree for reranking. This has currently been designed to work best with process reward models and other similar scoring functions, as opposed to the full flexibility we provide for `BestOfN` and `TreeSearch` that can e.g., harness grammar constraints or apply sharper scoring functions. As this interface is finalized and documented examples come together, it will be promoted to the fully supported `decoding.generators`.

### Sequantial Monte Carlo / Particle Filtering

[HFPPL](https://github.com/probcomp/hfppl) is a beautiful library for probabilistic programming with large language models, based on the work by [Lew et al., 2024](https://arxiv.org/abs/2306.03081) on _Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs_. It is on our roadmap to see how their underlying algorithms and infrastructure can be ported to our interface. In the meantime, check them out.

## Contributing

We welcome community contributions. Open an issue or a pull request if you see any ways to make `decoding` better.

To get started with development, this library supports an automated build of the dev and testing env using [GNU Make](https://www.gnu.org/software/make/).

```bash
# clone repo and install lib in current env
git clone git@github.com:benlipkin/decoding.git
cd decoding/
make env
```

Before opening a PR, make sure all tests are passing:

```bash
make tests
```

## Citation

```bibtex
@misc{decoding2024lipkin,
    author = {Lipkin, Benjamin},
    title = {Decoding: Composable inference algorithms with LLMs and programmable logic.},
    publisher = {GitHub},
    journal = {GitHub},
    howpublished = {\url{https://github.com/benlipkin/decoding}},
    year = 2024,
}
```
