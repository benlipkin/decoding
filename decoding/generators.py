"""
Generators are the user's highest-level interface to the decoding library.
By composing instances of `decoding.models.LanguageModel`, `decoding.scorers.Scorer`,
and control flow parameters that specify sync, stop, and search conditions, users can
implement a wide variety of decoding algorithms with very little code.

The `BestOfN` and `TreeSearch` generators are currently fully supported. There is also
experimental support for `RolloutTreeSearch` in the `decoding.experimental` module,
which supports a simple wrapper interface for a more standard Monte Carlo Tree Search
(MCTS) algorithm. It is also on the roadmap to bring twisted `SMC` to the
`decoding` library.

**NB**: The examples below are illustrative of the API, but not particularly useful.
See the [`examples`](https://github.com/benlipkin/decoding/tree/main/examples)
directory for more interesting examples.
"""

from collections.abc import Callable
from dataclasses import dataclass

from vllm.sampling_params import LogitsProcessor, SamplingParams
from vllm.transformers_utils.tokenizers import MistralTokenizer

from decoding.models import LanguageModel
from decoding.pmf import LogPMF, ScoredItem, sort_scored_items, topk_scored_items
from decoding.scorers import Scorer


@dataclass(frozen=True, kw_only=True)
class _SearchParams:
    n: int
    width: int
    max_steps: int
    stop_pass: Callable[[str], bool]
    stop_fail: Callable[[str], bool]


def BestOfN(  # noqa: PLR0913
    *,
    prompt: str,
    llm: LanguageModel,
    scorer: Scorer,
    n: int = 1,
    min_tokens: int = 0,
    max_tokens: int | None = None,
    stop_str: list[str] | str | None = None,
    stop_token_ids: list[int] | str | None = None,
    include_stop_str_in_output: bool = True,
    track_logprobs: bool = False,
    temperature: float = 1.0,
    logits_processors: list[LogitsProcessor] | None = None,
    seed: int | None = None,
) -> list[ScoredItem[str]]:
    """
    Generate `n` samples from the language model `llm` using the `scorer` to rank them.
    See the [`vLLM.SamplingParams`](https://docs.vllm.ai/en/latest/dev/sampling_params.html)
    docs to learn more about some of these parameters such as `logits_processors`.

    Args:
        prompt: The input prompt string.
        llm: The language model to generate samples from.
        scorer: The scorer to rank the samples.
        n: The number of samples to generate.
        min_tokens: The minimum number of tokens in each sample.
        max_tokens: The maximum number of tokens in each sample.
        stop_str: A string or list of strings that, if generated, will stop decoding.
        stop_token_ids: A list of token IDs that, if generated, will stop decoding.
            A string can also be passed, which will specify all token IDs that contain
            that substring.
        include_stop_str_in_output: Whether to include the stop string in the output.
        track_logprobs: Whether to track log probabilities. This comes at a performance
            cost, so it is off by default. In most cases, as you are alrady sampling
            from the model, you do not want to double count the probabilities in the
            scorer anyways.
        temperature: The temperature for sampling.
        logits_processors: A list of logits processors.
        seed: The random seed.

    Returns:
        A list of `decoding.pmf.ScoredItem` objects sorted by the `scorer`.

    Raises:
        ValueError: If any of the argument configurations are invalid.

    Examples:
        ```python
        from decoding.generators import BestOfN
        from decoding.models import LanguageModel
        from decoding.scorers import Scorer

        llm = LanguageModel.from_id("gpt2")
        scorer = Scorer.from_f_str_to_num(lambda x: -len(x))
        samples = BestOfN(
            prompt="The",
            llm=llm,
            scorer=scorer,
            n=20,
            stop_str=".",
            seed=42,
        )
        assert len(samples) == 20
        assert all(s.item.endswith(".") for s in samples)
        assert all(s.score == -len(s.item) for s in samples)
        assert samples[0].score >= samples[-1].score
        ```

    """
    sampling_params = SamplingParams(
        n=_guard_positive_int(n),
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        stop=stop_str,
        stop_token_ids=_prepare_token_ids(stop_token_ids, llm=llm),
        include_stop_str_in_output=include_stop_str_in_output,
        logprobs=_prepare_track_logprobs(track_logprobs),
        prompt_logprobs=_prepare_track_logprobs(track_logprobs),
        temperature=temperature,
        logits_processors=logits_processors,
        seed=seed,
        **_default_sampling_kwargs,  # type: ignore[reportArgumentType]
    )
    samples = _BestOfN([prompt], llm, scorer, sampling_params)
    return sort_scored_items(samples)


def TreeSearch(  # noqa: PLR0913
    *,
    prompt: str,
    llm: LanguageModel,
    step_scorer: Scorer,
    final_scorer: Scorer | None = None,
    stop_cond_pass: Callable[[str], bool],
    stop_cond_fail: Callable[[str], bool] | None = None,
    n: int = 1,
    beam_width: int = 1,
    beam_factor: int = 1,
    max_steps: int | None = None,
    min_tokens_per_step: int = 0,
    max_tokens_per_step: int | None = None,
    sync_str: list[str] | str | None = None,
    sync_token_ids: list[int] | str | None = None,
    include_sync_str_in_output: bool = True,
    track_logprobs: bool = False,
    temperature: float = 1.0,
    logits_processors: list[LogitsProcessor] | None = None,
    seed: int | None = None,
) -> list[ScoredItem[str]]:
    """
    Generate `n` samples from the language model `llm` using the `step_scorer` to
    rank them at each sync step and the `final_scorer` to rank the final beam.

    Args:
        prompt: The input prompt string.
        llm: The language model to generate samples from.
        step_scorer: The scorer to rank the samples at each sync step.
        final_scorer: The scorer to rank the final beam.
        stop_cond_pass: A function that returns `True` if the sample should pass.
            This stops the sample from being extended.
        stop_cond_fail: A function that returns `True` if the sample should fail.
            This filters the sample from the live beam.
        n: The number of passing samples to generate before returning.
        beam_width: The width of the beam. This is the number of samples to
            keep at each step.
        beam_factor: The branching factor of the beam. This is the number of
            new samples to generate from each live sample at each sync step.
        max_steps: The maximum number of sync steps to take.
        min_tokens_per_step: The minimum number of tokens in each step's extension.
        max_tokens_per_step: The maximum number of tokens in each step's extension.
        sync_str: A string or list of strings that, if generated, will stop extending
            each sample in the live beam and await scoring, ranking, and filtering.
        sync_token_ids: A list of token IDs that, if generated, will stop extending
            each sample in the live beam and await scoring, ranking, and filtering.
            A string can also be passed, which will specify all token IDs that contain
            that substring.
        include_sync_str_in_output: Whether to include the stop string in the output.
        track_logprobs: Whether to track log probabilities. This comes at a performance
            cost, so it is off by default. In most cases, as you are already sampling
            from the model, you do not want to double count the probabilities in the
            scorer anyways.
        temperature: The temperature for sampling.
        logits_processors: A list of logits processors.
            NB: This is applied within each step as opposed to globally.
        seed: The random seed.

    Returns:
        A list of `decoding.pmf.ScoredItem` objects sorted by the `final_scorer`.

    Raises:
        ValueError: If any of the argument configurations are invalid
        RuntimeError: if all live samples in the beam fail,
            or if max steps is reached before any samples pass.

    Examples:
        ```python
        from decoding.generators import TreeSearch
        from decoding.models import LanguageModel
        from decoding.pmf import ScoredItem
        from decoding.scorers import Scorer

        def f(x):
            if "." in x:
                x = x.split(".")[0] + "."
            return ScoredItem(item=x, score=-len(x))

        llm = LanguageModel.from_id("gpt2")
        scorer = Scorer.from_f_str_to_sample(f)
        samples = TreeSearch(
            prompt="The",
            sync_token_ids=" ",
            stop_cond_pass=lambda x: x.endswith("."),
            llm=llm,
            step_scorer=scorer,
            final_scorer=scorer,
            n=3,
            beam_width=50,
            beam_factor=5,
            seed=42,
        )
        assert len(samples) == 3
        assert all(s.item.endswith(".") for s in samples)
        assert all(s.score == -len(s.item) for s in samples)
        assert samples[0].score >= samples[-1].score
        ```

    """
    if final_scorer is None:
        final_scorer = step_scorer
    search_params = _SearchParams(
        n=_guard_positive_int(n),
        width=_guard_positive_int(beam_width),
        max_steps=_prepare_max_steps(max_steps),
        stop_pass=_prepare_stop(stop_cond_pass),
        stop_fail=_prepare_stop(stop_cond_fail),
    )
    _validate_search_params(search_params)
    sampling_params = SamplingParams(
        n=_guard_positive_int(beam_factor),
        min_tokens=min_tokens_per_step,
        max_tokens=max_tokens_per_step,
        stop=sync_str,
        stop_token_ids=_prepare_token_ids(sync_token_ids, llm=llm),
        include_stop_str_in_output=include_sync_str_in_output,
        logprobs=_prepare_track_logprobs(track_logprobs),
        prompt_logprobs=_prepare_track_logprobs(track_logprobs),
        temperature=temperature,
        logits_processors=logits_processors,
        seed=seed,
        **_default_sampling_kwargs,  # type: ignore[reportArgumentType]
    )
    samples = _TreeSearch([prompt], llm, step_scorer, search_params, sampling_params)
    return sort_scored_items(final_scorer(LogPMF.from_samples(samples)))


def _BestOfN(
    prompts: list[str],
    llm: LanguageModel,
    scorer: Scorer,
    sampling_params: SamplingParams,
) -> list[ScoredItem[str]]:
    return scorer(llm(prompts=prompts, params=sampling_params))


def _TreeSearch(
    prompts: list[str],
    llm: LanguageModel,
    scorer: Scorer,
    search_params: _SearchParams,
    sampling_params: SamplingParams,
) -> list[ScoredItem[str]]:
    beam = [ScoredItem(item=p, score=-float("inf")) for p in prompts]
    passing = []
    for _ in range(search_params.max_steps):
        stop_pass = [search_params.stop_pass(s.item) for s in beam]
        stop_fail = [search_params.stop_fail(s.item) for s in beam]
        passing = []
        prompts = []
        for sample, passed, failed in zip(beam, stop_pass, stop_fail, strict=True):
            if passed and not failed:
                passing.append(sample)
            elif not failed:
                prompts.append(sample.item)
            else:  # failed
                pass
        if len(passing) >= search_params.n:
            return passing
        if len(prompts) == 0:
            return _handle_failed_beam(passing)
        live = _BestOfN(prompts, llm, scorer, sampling_params)
        beam = passing + live
        if len(beam) > search_params.width:
            beam = topk_scored_items(beam, search_params.width)
    return _handle_maxsteps(passing)


def _prepare_token_ids(
    token_ids: list[int] | str | None, *, llm: LanguageModel
) -> list[int] | None:
    if isinstance(token_ids, str):
        return _get_token_ids_from_delimiter(llm=llm, delimiter=token_ids)
    return token_ids


def _get_token_ids_from_delimiter(*, llm: LanguageModel, delimiter: str) -> list[int]:
    _validate_delimiter(delimiter)
    tokenizer = llm.tokenizer
    if isinstance(tokenizer, MistralTokenizer):
        msg = "vLLM Mistral tokenizer does not currently support `batch_decode`."
        raise NotImplementedError(msg)
    tokens = list(tokenizer.get_vocab().values())
    strs = tokenizer.batch_decode(tokens)
    return [tokens[i] for i, s in enumerate(strs) if delimiter in s]


def _validate_search_params(params: _SearchParams) -> None:
    if params.n > params.width:
        msg = "`beam_width` cannot be less than `n`."
        raise ValueError(msg)


def _validate_delimiter(delimiter: str) -> None:
    if len(delimiter) != 1:
        msg = f"Delimiter must be a single character, got: {delimiter}."
        raise ValueError(msg)


def _prepare_stop(
    stop: Callable[[str], bool] | None,
) -> Callable[[str], bool]:
    if stop is None:

        def _dont_stop(_: str) -> bool:
            return False

        return _dont_stop
    return stop


def _prepare_max_steps(max_steps: int | None) -> int:
    if max_steps is None:
        return 2**32
    return _guard_positive_int(max_steps)


def _prepare_track_logprobs(track_logprobs: bool) -> int | None:  # noqa: FBT001
    return 0 if track_logprobs else None


def _guard_positive_int(n: int) -> int:
    if n < 1:
        msg = f"Expected a positive integer, got: {n}."
        raise ValueError(msg)
    return n


def _handle_failed_beam(passing: list[ScoredItem[str]]) -> list[ScoredItem[str]]:
    if len(passing) == 0:
        msg = "All live samples failed before any passed stop conditions."
        msg += " Check compatibility of stop conditions or expand search."
        raise RuntimeError(msg)
    import warnings

    msg = "All live samples failed before completing search,"
    msg += " but some completed samples have already passed stopping conditions."
    msg += " Returning available passing samples."
    warnings.warn(msg, stacklevel=2)
    return passing


def _handle_maxsteps(passing: list[ScoredItem[str]]) -> list[ScoredItem[str]]:
    if len(passing) == 0:
        msg = "Max steps reached, and no samples passed stop conditions."
        raise RuntimeError(msg)
    import warnings

    msg = "Max steps reached before completing search,"
    msg += "but some samples have already passed stopping conditions."
    msg += " Returning available passing samples."
    warnings.warn(msg, stacklevel=2)
    return passing


_default_sampling_kwargs = {
    "detokenize": True,
    "ignore_eos": False,
    "truncate_prompt_tokens": None,
}
