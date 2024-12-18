"""
Language Model interface for text generation and scoring.

This module wraps the vLLM library to provide a simple interface via the
`LanguageModel` class. The class provides methods for conditionally generating
strings with `LanguageModel.generate` and scoring strings with `LanguageModel.surprise`.
An easy constructor is also provided to load a model by its Hugging Face model ID
and specify optional parameters for memory management, KV caching, scheduling policy,
quantization, LORA, speculative decoding, and many other settings.
"""

# ruff: noqa: E402

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard, Unpack

import jax.numpy as jnp

from decoding.pmf import LogPMF
from decoding.types import FVX

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
_logger.info("Importing vLLM: This may take a moment...")

from vllm import LLM, EngineArgs, SamplingParams
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer


@dataclass(frozen=True, kw_only=True)
class LanguageModel:
    """
    vLLM-backed Language Model interface for text generation and scoring.
    """

    _llm: LLM

    @property
    def tokenizer(self) -> AnyTokenizer:
        """
        Return the tokenizer used by the language model.
        """
        return self._llm.get_tokenizer()

    def __call__(
        self, *, prompts: Sequence[str], params: SamplingParams
    ) -> LogPMF[str]:
        """
        `__call__` is an alias for `generate`.
        """
        return self.generate(prompts=prompts, params=params)

    def generate(
        self, *, prompts: Sequence[str], params: SamplingParams
    ) -> LogPMF[str]:
        """
        Generate text conditioned on the given prompts.

        Args:
            prompts: A batch of prompts.
            params: An instance of
                [`vllm.SamplingParams`](https://docs.vllm.ai/en/latest/dev/sampling_params.html)
                for sampling. See the vLLM documentation for more details.

        Returns:
            A `decoding.pmf.LogPMF` instance containing the distribution of
            generated text. The categories of this distribution are the strings
            generated by the model. For `k` prompts with `SamplingParams.n=n`, the
            distribution will have `n*k` categories. The log-probs of this distribution
            reflect the *normalized* log-probabilities of this batch of strings,
            calculated from the model's logits. If the `vllm.SamplingParams` are set
            such as to not track log-probs, the log-probs will be uniform, i.e.,
            `-log(n*k)`.

        Raises:
            ValueError: If the `params` are not valid.

        Examples:
            ```python
            import jax.nn as jnn
            from vllm import SamplingParams
            from decoding.models import LanguageModel

            llm = LanguageModel.from_id("gpt2")
            prompts = ["The quick brown fox", "The slow white rabbit"]
            params = SamplingParams(n=5)

            d = llm.generate(prompts=prompts, params=params)
            assert jnn.logsumexp(d.logp) == 0.0
            assert len(d.items) == 10
            ```

        """
        if _check_track_logprobs(params):
            return self._generate_w_logprobs(prompts=list(prompts), params=params)
        return self._generate_wo_logprobs(prompts=list(prompts), params=params)

    def _generate_w_logprobs(
        self, *, prompts: list[str], params: SamplingParams
    ) -> LogPMF[str]:
        _validate_sampling_params(params, track_logprobs=True)
        responses = self._llm.generate(prompts, params)
        items, logits = [], []
        for response in responses:
            prompt_text = _guard_prompt_text(response.prompt)
            prompt_logps = _parse_prompt_logps(response, self.tokenizer)
            prompt_logp = sum(next(iter(x.values())).logprob for x in prompt_logps)
            for output in response.outputs:
                output_logp = _guard_output_logp(output.cumulative_logprob)
                items.append(prompt_text + output.text)
                logits.append(prompt_logp + output_logp)
        return LogPMF.from_logits(logits=jnp.asarray(logits), items=items)

    def _generate_wo_logprobs(
        self, *, prompts: list[str], params: SamplingParams
    ) -> LogPMF[str]:
        _validate_sampling_params(params, track_logprobs=False)
        responses = self._llm.generate(prompts, params)
        samples = []
        for response in responses:
            prompt_text = _guard_prompt_text(response.prompt)
            samples.extend([prompt_text + o.text for o in response.outputs])
        return LogPMF.from_samples(samples)

    def surprise(self, *, contexts: Sequence[str], queries: Sequence[str]) -> FVX:
        """
        Calculate the surprisal of queries given contexts.

        Args:
            contexts: A batch of contexts.
            queries: A batch of queries.

        Returns:
            A JAX 1D array of surprisals for each query given its context.
            The surprisal is calculated as the negative log-probability
            of the query given the context.

        Raises:
            ValueError: If the queries are empty, or if the contexts and queries do not
            match in length.

        Examples:
            ```python
            import jax.numpy as jnp
            from decoding.models import LanguageModel

            llm = LanguageModel.from_id("gpt2")
            contexts = ["The quick brown fox", "The slow white rabbit"]
            queries = [" jumps over the lazy dog", " hops over the sleeping cat"]

            s = llm.surprise(contexts=contexts, queries=queries)
            assert jnp.all(s >= 0.0)
            assert s.shape == (2,)
            ```

        """
        _validate_queries_nonempty(queries)
        _validate_matched_contexts_queries(contexts, queries)
        prompts = [f"{c}{q}" for c, q in zip(contexts, queries, strict=True)]
        params = SamplingParams(prompt_logprobs=0, max_tokens=1)
        responses = self._llm.generate(prompts, params)
        surprisals = []
        for context, response in zip(contexts, responses, strict=True):
            prompt_text = _guard_prompt_text(response.prompt)
            prompt_logps = _parse_prompt_logps(response, self.tokenizer)
            marker, final, tracker, surprise = len(context), len(prompt_text), 0, 0
            for item in prompt_logps:
                token = next(iter(item.values()))
                tracker += len(_guard_decoded_token(token.decoded_token))
                if tracker > marker:
                    surprise -= token.logprob
            _validate_alignment(tracker, final)
            surprisals.append(surprise)
        return jnp.asarray(surprisals)

    @classmethod
    def from_id(
        cls,
        model_id: str,
        **model_kwargs: Unpack[EngineArgs],  # type: ignore[reportGeneralTypeIssues]
    ) -> "LanguageModel":
        """
        Load a language model by its Hugging Face model ID.

        Args:
            model_id: The Hugging Face model ID.
            **model_kwargs: Optional parameters for the model constructor. These are
                passed to the [`vllm.LLM`](https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html)
                constructor, and through there to [`vllm.EngineArgs`](https://docs.vllm.ai/en/stable/models/engine_args.html).
                Check the linked vLLM documentation for more details on what parameters
                are available.

        Returns:
            A `LanguageModel` instance.

        Examples:
            ```python
            from decoding.models import LanguageModel

            llm = LanguageModel.from_id(
                "gpt2", gpu_memory_utilization=0.5, enable_prefix_caching=True
            )
            assert llm.tokenizer.name_or_path == "gpt2"
            ```

        """
        llm = LLM(model_id, **model_kwargs)
        return cls(_llm=llm)


def _check_track_logprobs(params: SamplingParams) -> bool:
    return params.logprobs is not None


def _parse_prompt_logps(
    response: RequestOutput, tokenizer: AnyTokenizer
) -> list[dict[int, Logprob]]:
    if response.prompt_token_ids is None:
        msg = "Prompt token ids should not be None"
        raise ValueError(msg)
    first_token_id = response.prompt_token_ids[0]
    first_token_str = tokenizer.convert_ids_to_tokens([first_token_id])[0]
    return _prepare_prompt_logp(
        response.prompt_logprobs, first_token_id, first_token_str
    )


def _prepare_prompt_logp(
    x: list[dict[int, Logprob] | None] | None,
    first_token_id: int,
    first_token_str: str,
) -> list[dict[int, Logprob]]:
    def _is_valid_logp(
        x: list[dict[int, Logprob] | None],
    ) -> TypeGuard[list[dict[int, Logprob]]]:
        return len(x) > 0 and all(y is not None and len(y.keys()) == 1 for y in x)

    if x is None:
        msg = "Prompt logprobs should not be None"
        raise ValueError(msg)
    x = [{first_token_id: Logprob(0, decoded_token=first_token_str)}] + x[1:]
    if _is_valid_logp(x):
        return x
    msg = "Error parsing prompt logprobs"
    raise ValueError(msg)


def _guard_prompt_text(x: PromptType | None) -> str:
    if isinstance(x, str):  # `PromptType` is a union type of `str` and other types
        return x
    msg = "Prompt text should be a string"
    raise ValueError(msg)


def _guard_output_logp(x: float | None) -> float:
    if x is None:
        msg = "Output logprob should not be None"
        raise ValueError(msg)
    return x


def _guard_decoded_token(x: str | None) -> str:
    if x is None:
        msg = "Decoded token should not be None"
        raise ValueError(msg)
    return x


def _validate_sampling_params(params: SamplingParams, *, track_logprobs: bool) -> None:
    correct_logp_status = 0 if track_logprobs else None
    if not all(
        (
            params.detokenize,
            params.logprobs == correct_logp_status,
            params.prompt_logprobs == correct_logp_status,
        )
    ):
        msg = "Error in sampling parameters: `detokenize` must be True, and"
        msg += " `logprobs` & `prompt_logprobs` must be `0` to track else `None`"
        raise ValueError(msg)


def _validate_queries_nonempty(queries: Sequence[str]) -> None:
    if not all(queries):
        msg = "Queries must be non-empty"
        raise ValueError(msg)


def _validate_matched_contexts_queries(
    contexts: Sequence[str], queries: Sequence[str]
) -> None:
    if len(contexts) != len(queries):
        msg = "Contexts and queries must have the same length"
        raise ValueError(msg)


def _validate_alignment(tracker: int, final: int) -> None:
    if tracker != final:
        msg = "Error aligning tokens while calculating surprise"
        raise ValueError(msg)
