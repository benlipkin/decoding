import pytest

from decoding.generators import BestOfN, TreeSearch
from decoding.models import LanguageModel
from decoding.pmf import ScoredItem
from decoding.scorers import Scorer

llm = LanguageModel.from_id("EleutherAI/pythia-70m", gpu_memory_utilization=0.2)


def test_bestofn() -> None:
    start = "The"

    def score(s: str) -> int:
        return -len(s)

    scorer = Scorer.from_f_str_to_num(score)
    sentences = {}
    for n in [1, 10, 100]:
        samples = BestOfN(
            llm=llm, scorer=scorer, prompt="The", stop_str=".", n=n, seed=0
        )
        sentences[n] = samples[0].item
    assert all(s.startswith("The") for s in sentences.values())
    assert all(s.endswith(".") for s in sentences.values())
    assert score(sentences[100]) > score(sentences[10]) > score(sentences[1])

    msg = "Delimiter must be a single character"
    with pytest.raises(ValueError, match=msg):
        BestOfN(llm=llm, scorer=scorer, prompt=start, stop_token_ids=start)
    msg = "Expected a positive integer"
    with pytest.raises(ValueError, match=msg):
        BestOfN(llm=llm, scorer=scorer, prompt=start, n=-1)


def test_treesearch_basic() -> None:
    start = "The"
    delim = " "
    end = "."

    def stop(s: str) -> bool:
        return end in s

    def score(s: str) -> int:
        if stop(s):
            return 1
        return -len(s)

    scorer = Scorer.from_f_str_to_num(score)
    sentence = TreeSearch(
        llm=llm,
        step_scorer=scorer,
        prompt=start,
        sync_token_ids=delim,
        stop_cond_pass=stop,
        n=1,
        beam_width=20,
        beam_factor=2,
        seed=0,
    )[0].item
    assert sentence.startswith(start)
    assert end in sentence

    def run_w_kwargs(n: int = 1, beam_width: int = 1, beam_factor: int = 1) -> None:
        TreeSearch(
            llm=llm,
            step_scorer=scorer,
            prompt=start,
            stop_cond_pass=stop,
            n=n,
            beam_factor=beam_factor,
            beam_width=beam_width,
            seed=0,
        )

    msg = "Expected a positive integer"
    with pytest.raises(ValueError, match=msg):
        run_w_kwargs(n=0)
    with pytest.raises(ValueError, match=msg):
        run_w_kwargs(beam_width=0)
    with pytest.raises(ValueError, match=msg):
        run_w_kwargs(beam_factor=0)
    msg = "`beam_width` cannot be less than `n`"
    with pytest.raises(ValueError, match=msg):
        run_w_kwargs(n=10, beam_width=5)


def test_treesearch_step() -> None:
    start = "The"
    delim = " "
    end = "."

    def stop(s: str) -> bool:
        return end in s

    def score_step(s: str) -> int:
        if stop(s):
            return 1
        ws = s.split(delim)
        check_words = 2
        if len(ws) < check_words:
            return -len(s)
        return -(len(ws[-2]) + len(ws[-1]))

    def score_final(s: str) -> int:
        return -len(s)

    step_scorer = Scorer.from_f_str_to_num(score_step)
    final_scorer = Scorer.from_f_str_to_num(score_final)
    sentence = TreeSearch(
        llm=llm,
        step_scorer=step_scorer,
        final_scorer=final_scorer,
        prompt=start,
        sync_token_ids=delim,
        stop_cond_pass=stop,
        n=1,
        beam_width=20,
        beam_factor=2,
        seed=0,
    )[0].item
    assert sentence.startswith(start)
    assert end in sentence


def test_treesearch_fail() -> None:
    max_len_constraint = 20
    start = "The"
    delim = " "
    end = "."

    def stop(s: str) -> bool:
        return end in s

    def score(s: str) -> int:
        if stop(s):
            return 1
        return -len(s)

    def fail(s: str) -> bool:
        return len(s) > max_len_constraint

    scorer = Scorer.from_f_str_to_num(score)

    def beam_search(n: int, beam_width: int, beam_factor: int) -> list[ScoredItem[str]]:
        return TreeSearch(
            llm=llm,
            step_scorer=scorer,
            prompt=start,
            sync_token_ids=delim,
            stop_cond_pass=stop,
            stop_cond_fail=fail,
            n=n,
            beam_width=beam_width,
            beam_factor=beam_factor,
            seed=0,
        )

    out = beam_search(1, 30, 6)
    sentence = out[0].item
    assert len(sentence) <= max_len_constraint
    assert sentence.startswith(start)
    assert end in sentence

    n_requested = 5
    msg = "All live samples failed before completing search"
    with pytest.warns(UserWarning, match=msg):
        out = beam_search(n_requested, 30, 6)
    assert 0 < len(out) < n_requested
    assert all(len(s.item) <= max_len_constraint for s in out)
    assert all(s.item.startswith(start) for s in out)
    assert all(end in s.item for s in out)

    msg = "All live samples failed before any passed stop conditions"
    with pytest.raises(RuntimeError, match=msg):
        beam_search(1, 30, 2)


def test_treesearch_maxsteps() -> None:
    start = "The"
    delim = " "
    end = "."

    def stop(s: str) -> bool:
        return end in s

    def score(s: str) -> int:
        if stop(s):
            return 1
        return -len(s)

    scorer = Scorer.from_f_str_to_num(score)
    n_requested = 3

    def beam_search(max_steps: int) -> list[ScoredItem[str]]:
        return TreeSearch(
            llm=llm,
            step_scorer=scorer,
            prompt=start,
            sync_token_ids=delim,
            stop_cond_pass=stop,
            max_steps=max_steps,
            n=n_requested,
            beam_width=30,
            beam_factor=6,
            seed=0,
        )

    out = beam_search(5)
    sentence = out[0].item
    assert len(out) == n_requested
    assert sentence.startswith(start)
    assert end in sentence

    msg = "Max steps reached before completing search"
    with pytest.warns(UserWarning, match=msg):
        out = beam_search(3)
    assert all(s.item.startswith(start) for s in out)
    assert all(end in s.item for s in out)

    msg = "Max steps reached, and no samples passed stop conditions"
    with pytest.raises(RuntimeError, match=msg):
        beam_search(1)

    msg = "Expected a positive integer"
    with pytest.raises(ValueError, match=msg):
        beam_search(-1)
