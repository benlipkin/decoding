# ruff: noqa: D100 D103

from collections.abc import Callable

from vllm.sampling_params import LogitsProcessor, SamplingParams

from decoding.generators import (
    _BeamSearch,  # type: ignore[reportPrivateUsage]
    _default_sampling_kwargs,  # type: ignore[reportPrivateUsage]
    _guard_positive_int,  # type: ignore[reportPrivateUsage]
    _prepare_max_steps,  # type: ignore[reportPrivateUsage]
    _prepare_stop,  # type: ignore[reportPrivateUsage]
    _prepare_token_ids,  # type: ignore[reportPrivateUsage]
    _prepare_track_logprobs,  # type: ignore[reportPrivateUsage]
    _SearchParams,  # type: ignore[reportPrivateUsage]
)
from decoding.models import LanguageModel
from decoding.pmf import CategoricalLogPMF, Sample, make_samples, sort_samples
from decoding.scorers import Scorer


def MCTS(  # noqa: PLR0913
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
    sync_token_ids: list[int] | None = None,
    include_sync_str_in_output: bool = True,
    track_logprobs: bool = False,
    temperature: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    logits_processors: list[LogitsProcessor] | None = None,
    seed: int | None = None,
) -> list[Sample[str]]:
    if final_scorer is None:
        final_scorer = step_scorer
    search_params = _SearchParams(
        n=_guard_positive_int(n),
        width=_guard_positive_int(beam_width),
        max_steps=_prepare_max_steps(max_steps),
        stop_pass=_prepare_stop(stop_cond_pass),
        stop_fail=_prepare_stop(stop_cond_fail),
    )
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
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        logits_processors=logits_processors,
        seed=seed,
        **_default_sampling_kwargs,  # type: ignore[reportArgumentType]
    )
    samples = _MCTS([prompt], llm, step_scorer, search_params, sampling_params)
    return sort_samples(final_scorer(CategoricalLogPMF.from_samples(samples)))


def _MCTS(
    prompts: list[str],
    llm: LanguageModel,
    scorer: Scorer,
    search_params: _SearchParams,
    sampling_params: SamplingParams,
) -> list[Sample[str]]:
    def f(d: CategoricalLogPMF[str]) -> list[Sample[str]]:
        _search_params = _SearchParams(
            n=1,
            width=search_params.width,
            max_steps=search_params.max_steps,
            stop_pass=search_params.stop_pass,
            stop_fail=search_params.stop_fail,
        )
        prompts = list(d.cats)
        utilities = []
        for prompt in prompts:
            try:
                samples = _BeamSearch(
                    [prompt], llm, scorer, _search_params, sampling_params
                )
            except ValueError:
                samples = [Sample(item=prompt, utility=-float("inf"))]
            utilities.append(samples[0].utility)
        return make_samples(prompts, utilities)

    _scorer = Scorer.from_f_catlogpmf_to_batch_sample(f)
    return _BeamSearch(prompts, llm, _scorer, search_params, sampling_params)
