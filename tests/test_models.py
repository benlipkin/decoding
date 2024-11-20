import pytest
from vllm import SamplingParams

from decoding.models import LanguageModel

llm = LanguageModel.from_id("EleutherAI/pythia-70m", gpu_memory_utilization=0.2)


def test_llm_generate() -> None:
    sp = SamplingParams(detokenize=True, logprobs=None, prompt_logprobs=None, n=2)
    prompts = ["The"] * 3
    outputs = llm(prompts=prompts, params=sp)
    assert len(outputs.items) == len(prompts) * sp.n
    assert len(set(outputs.logp.tolist())) == 1

    _sp = sp.clone()
    _sp.logprobs = 0
    _sp.prompt_logprobs = 0
    outputs = llm(prompts=prompts, params=_sp)
    assert len(outputs.items) == len(prompts) * sp.n
    assert len(set(outputs.logp.tolist())) == len(prompts) * sp.n

    sp.detokenize = False
    with pytest.raises(ValueError, match="Error in sampling parameters:"):
        llm(prompts=prompts, params=sp)


def test_llm_surprise() -> None:
    contexts = ["The"] * 2
    queries = [" quick", "asdf"]
    surprisals = llm.surprise(contexts=contexts, queries=queries)
    assert all(s > 0 for s in surprisals)
    assert surprisals[0] < surprisals[1]

    c1 = ["The "]
    q1 = ["quick"]
    c2 = ["The q"]
    q2 = ["uick"]
    s1 = llm.surprise(contexts=c1, queries=q1)
    s2 = llm.surprise(contexts=c2, queries=q2)
    assert s1[0] == s2[0]  # measure surprise at decision boundary if mid-token
    # in theory should marginalize over all remaining possible tokens in trie
    # but this approximation is good enough for the library's goals

    with pytest.raises(ValueError, match="Queries must be non-empty"):
        llm.surprise(contexts=[""], queries=[""])

    with pytest.raises(
        ValueError, match="Contexts and queries must have the same length"
    ):
        llm.surprise(contexts=["a"], queries=["a", "a"])
