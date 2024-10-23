# Tutorial

Let's walk through a few examples of how to use `decoding` to design custom inference algorithms that solve fun tasks.

## 1. Solve a structured generation task by sampling from a small constrained LM, and reranking with a large unconstrained LM

Let's say you have a task that requires generating structured outputs, such as formalizing a series of sentences into mathematical expressions. One successful pattern here is to draw constrained samples from a small LM, and then re-score them using an unconstrained big LM. Check out how simple this pattern is:

```python
import re
from collections.abc import Sequence

from vllm.model_executor.guided_decoding.outlines_logits_processors import RegexLogitsProcessor

from decoding.generators import BestOfN
from decoding.models import LanguageModel
from decoding.scorers import Scorer

# load your small and big models
small_lm = LanguageModel.from_id("allenai/OLMo-1B-hf", gpu_memory_utilization=0.2)
large_lm = LanguageModel.from_id("allenai/OLMo-7B-hf", gpu_memory_utilization=0.6)

# write a scoring function and construct a scoring object
def score_fn(prompts: list[str]) -> list[float]:
    contexts = [""] * len(prompts)
    logps = -large_lm.surprise(contexts=contexts, queries=prompts)
    return logps

scorer = Scorer.from_f_batch_str_to_batch_num(score_fn)
# we support several constructors for scorers so you can decide if you'd like to:
# - handle batching yourself or delegate it
# - access sequence probabilities in your scoring or ignore them
# - leave generations uninterrupted or modify them, 
#   e.g., by executing code and returning output or even backtracking on parse failures
# here, we're just offering to return an array of numbers for a sequence of strs 
# and the library is handling the rest


# let's write our prompt and decide on our desired output pattern
prompt = """Translate the following sentences into arithmetic expressions:

Q: The difference between 5 and 2
A:"""

pattern = r" \d+ [\+\-] \d+\n"
processors = [RegexLogitsProcessor(pattern, small_lm.tokenizer)]

# and let's wrap this up with a `BestOfN` generator that will:
# - sample n generations from the small model
# - return them re-reranked by the big model
# we'll just take the top output, and return its value
def bon(prompt: str, n: int) -> str:
    return BestOfN(
        prompt=prompt,
        llm=small_lm,
        scorer=scorer,
        n=n,
        stop_str="\n",
        logits_processors=processors,
        seed=42,
    )[0].item

# let's run with n=5 and see what we get
out = bon(prompt, n=5)
expr = re.findall(pattern, out)[0].strip()
assert expr == "5 - 2"

# looks good
```

Note how we solved this task zero-shot, with perfect formatting, and using only a 1B model as the generator, because we implemented a well-designed inference strategy.

In this example, we introduced a few of the relevant `decoding` idioms. We imported a `LanguageModel`, a `Scorer`, and `BestOfN` for our generation algorithm.

The `LanguageModel` class wraps `vLLM`, supporting specification of `gpu_memory_utilization` among other relevant arguments, and keeping text generation fast. See [`decoding.models`](https://benlipkin.github.io/decoding/decoding/models.html) docs to learn more about the LM interface.

The `Scorer` class orchestrates the calculation of utilities for each sample generated from the LM and returns them in order. As noted above, a scorer can be constructed from a number of different functions, depending on what the user wants to control vs what they want to delegate. See [`decoding.scorers`](https://benlipkin.github.io/decoding/decoding/scorers.html) docs to learn more about the scoring interface.

The `BestOfN` function is the user's interface to the generation process. Here they may specify the `stop_str`, add `logits_processors`, or even tweak the `temperature`. See [`decoding.generators`](https://benlipkin.github.io/decoding/decoding/generators.html) docs to learn more about the generator interfaces.

Now, let's check a more involved example.

## 2. Solve a theorem proving task by running `TreeSearch` line-by-line, backtracking on syntax-errors, and collapsing the final beam with `SelfConsistency`.

```python
# first, install `nltk` to run this example
from nltk.inference import TableauProver
from nltk.sem import Expression
from nltk.sem.logic import LogicalExpressionException, LogicParser

from decoding.generators import TreeSearch
from decoding.estimators import SelfConsistency
from decoding.models import LanguageModel
from decoding.pmf import CategoricalLogPMF, Sample
from decoding.scorers import Scorer

# here's our prompt for the problem we'd like solved
prompt = """Formalize the following sentences into first-order logic:

English:
P: Socrates is a man.
P: All men are mortal.
C: Is Socrates mortal?

FOL:
P: man(socrates)
P: all x.(man(x) -> mortal(x))
C: mortal(socrates)

English:
P: All rectangles have four sides.
P: All four-sided things are quadrilaterals.
C: Is a rectangle a quadrilateral?

FOL:
P:"""

# let's load our LM, parser, and prover
llm = LanguageModel.from_id("microsoft/Phi-3-mini-4k-instruct")
expr = Expression.fromstring
parser = LogicParser()
prover = TableauProver()

# let's specify our conditions for syncing and stopping particles
sync_str="\n"
stop_str="\n\n"

def stop_pass(s: str) -> bool:
    return s.endswith(stop_str)

# let's specify how to score particles at each step
# note that compared to the previous example,
# here instead of simply returning a float, 
# we're returning a `Sample`: a str with an associated utility
# this will allow us to modify the state of the string
def step_score_fn(s: str) -> Sample[str]:
    if stop_pass(s):
        return Sample(item=s, utility=float("inf"))
    lines = s.strip().split("\n")
    last_line = lines[-1]
    if last_line.startswith(("P:", "C:")):
        stmt = last_line[2:]
        try:
            parser.parse(stmt)
            return Sample(item=s, utility=len(lines))
        except LogicalExpressionException:
            pass
    backtrack = "\n".join(lines[:-1]) + "\n"
    return Sample(item=backtrack, utility=len(lines) - 1)
# the logic above is as follows:
# - if a string passes the stop condition, set utility high to keep it
# - for the strings that are not done, try to parse the last line
# - if is parses, keep it and update the utility to the number of passing lines
# - if it fails, backtrack the string to the last passing line
# using a very simple (~10 line) step function, 
# we've implemented a backtracking tree search algorithm
# that provides process supervision on syntactic validity

# let's construct a scorer object
# here, given that our scoring proceeds independently for each str,
# we can parallelize it over a batch
step_scorer = Scorer.from_f_str_to_sample(step_score_fn, parallelize=True)

# now let's specify our final score function 
# to resolve the beam of passing particles
def final_score_fn(gens: CategoricalLogPMF[str]) -> list[Sample[str]]:
    def postproc(gen: str) -> str:
        try:
            new = gen[len(prompt) - 2 :]
            stmts = new.split("\n")
            premises = [expr(s[2:].strip()) for s in stmts if s.startswith("P:")]
            conclusions = [expr(s[2:].strip()) for s in stmts if s.startswith("C:")]
            if len(premises) == 0 or len(conclusions) != 1:
                return "Error"
            if prover.prove(conclusions[0], premises):
                return "True"
            if prover.prove(conclusions[0].negate(), premises):
                return "False"
            return "Unknown"
        except Exception:
            return "Error"

    def filt(s: str) -> bool:
        return s != "Error"

    return SelfConsistency(gens, postproc=postproc, filt=filt, parallelize=True)
# we are going to use a self-consistency algorithm that will: 
# - apply a post-processor on each generated str
# - filter those outputs
# - and majority vote on the remainder
# the postprocessor will:
# - extract the premises and conclusion
# - use a theorem prover to return a label
# the filter will remove all error samples
# and we'll return the best sample

# here we'll use this to construct a scorer
final_scorer = Scorer.from_f_catlogpmf_to_batch_sample(final_score_fn)
# note that `gens` here is an instance of the `CategoricalLogPMF` class,
# a categorical log probability mass function
# one of the underlying building blocks of this library
# we don't need to work with it, as we saw in the first example
# but if we'd like to use the string probabilities to weight the `SelfConsistency`,
# we can access it

# finally, let's wrap this all up in a `TreeSearch` generator
def run(prompt: str) -> list[Sample[str]]:
    return TreeSearch(
        prompt=prompt,
        llm=llm,
        step_scorer=step_scorer,
        final_scorer=final_scorer,
        stop_cond_pass=stop_pass,
        sync_str=sync_str,
        n=10, # number of complete samples to search for
        beam_width=25, # size of beam to maintain
        beam_factor=5, # number of ways to split each particle at each step
        seed=42,
    )


# run it, and extract the top output
out = run(prompt)
label = out[0].item
assert label == "True"

# looks good
```

Note that here again, we relied on a small 3B model and used a well-designed algorithm with process supervision and backtracking to solve this task.

This example introduced a few more important patterns. 

We imported a `LanguageModel`, a `Scorer`, and a generator (here `TreeSearch`) again, as well as `SelfConsistency`, an estimator.

The `decoding.estimators` module contains methods for reducing probability distributions, including a simple `MAP` estimator and building blocks for various `MBR` (minimum Bayes risk) estimators that accept arbitrary user-defined utility functions. In addition to the standard fully assymetric $O(n^2)$ `MBR` variant, we provide variants that shave off a factor of $2$ in case of symmetry with `commutativeMBR`, or an $O(n)$ `linearMBR` algorithm when utilities can be computed independently on a per-element basis. See the [`decoding.estimators`](https://benlipkin.github.io/decoding/decoding/estimators.html) docs to learn more.

In addition to the `TreeSearch` generator for general cases where we want synchronous rescoring, those underlying components support additional increasingly specific algorithms such as a powerful Monte Carlo Tree Search (MCTS) variant, via the `RolloutTreeSearch` generator, which can be found in `decoding.experimental`. This algorithm is based off of `TreeSearch` using a rollout mechanism inside scoring, but the interface is still under development. It will be migrated to the core library when it is stable.

## 3. What next?

We saw above some simple patterns for how to use the `decoding` library to solve some interesting tasks. We learned that with the right set of abstractions, it can be simple to implement powerful and effective inference algorithms using very little code. This style of workflow is what makes the library so powerful: the idea to implementation loop is tight, and the code factors intuitively. Try throwing together your own example now, and get creative with it. Check out the [docs](https://benlipkin.github.io/decoding) to learn more about the API.