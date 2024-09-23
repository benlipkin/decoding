import jax.numpy as jnp
import jax.random as jr
import pytest

from decoding.pmf import CategoricalLogPMF
from decoding.samplers import greedy, minp, random, topk, topp
from decoding.utils import getkey, logsoftmax

_N = 10_000


def _make_char_unigram() -> CategoricalLogPMF[str]:
    chars = "abcdefghijklmnopqrstuvwxyz"
    logprobs = logsoftmax(jr.normal(getkey(), (len(chars),)))
    return CategoricalLogPMF(logp=logprobs, cats=chars)


def test_greedy() -> None:
    p = jnp.log(jnp.asarray([0, 0.5, 0, 0.5]))
    d_i = CategoricalLogPMF(logp=p, cats=[0, 1, 2, 3])
    assert greedy(d_i) == 1

    d_c = _make_char_unigram()
    assert greedy(d_c) == d_c.cats[d_c.logp.argmax().item()]

    with pytest.raises(ValueError, match="n must be at least 1"):
        greedy(d_c, n=0)


def test_random() -> None:
    p = jnp.log(jnp.asarray([0.5, 0.5]))
    d_i = CategoricalLogPMF(logp=p, cats=[0, 1])

    samples = jnp.asarray(random(d_i, n=_N))
    assert jnp.isclose(jnp.mean(samples), 0.5, atol=0.1)

    with pytest.raises(ValueError, match="n must be at least 1"):
        random(d_i, n=0)


def test_topk() -> None:
    p = jnp.log(jnp.asarray([0.3, 0.3, 0.2, 0.2]))
    d_i = CategoricalLogPMF(logp=p, cats=[0, 1, 2, 3])

    samples = jnp.asarray(topk(d_i, k=2, n=_N))
    assert jnp.isclose(jnp.mean(samples), 0.5, atol=0.1)

    topk_samples = jnp.asarray(topk(d_i, k=len(d_i.cats), n=_N))
    random_samples = jnp.asarray(random(d_i, n=_N))
    assert jnp.isclose(jnp.mean(topk_samples), jnp.mean(random_samples), atol=0.1)

    d_c = _make_char_unigram()
    assert topk(d_c, k=1) == greedy(d_c)

    with pytest.raises(ValueError, match="n must be at least 1"):
        topk(d_c, k=1, n=0)


def test_topp() -> None:
    p = jnp.log(jnp.asarray([0.3, 0.3, 0.2, 0.2]))
    d_i = CategoricalLogPMF(logp=p, cats=[0, 1, 2, 3])

    samples = jnp.asarray(topp(d_i, p=0.5, n=_N))
    assert jnp.mean(samples) == 0
    samples = jnp.asarray(topp(d_i, p=0.6, n=_N))
    assert jnp.isclose(jnp.mean(samples), 0.5, atol=0.1)
    samples = jnp.asarray(topp(d_i, p=0.7, n=_N))
    assert jnp.isclose(jnp.mean(samples), 0.5, atol=0.1)
    samples = jnp.asarray(topp(d_i, p=1.0, n=_N))
    assert jnp.isclose(jnp.mean(samples), 1.3, atol=0.1)

    topp_samples = jnp.asarray(topp(d_i, p=1, n=_N))
    random_samples = jnp.asarray(random(d_i, n=_N))
    assert jnp.isclose(jnp.mean(topp_samples), jnp.mean(random_samples), atol=0.1)

    d_c = _make_char_unigram()
    assert topp(d_c, p=0) == greedy(d_c)

    with pytest.raises(ValueError, match="n must be at least 1"):
        topp(d_c, p=0.5, n=0)
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        topp(d_c, p=-0.1)


def test_minp() -> None:
    p = jnp.log(jnp.asarray([0.3, 0.3, 0.2, 0.2]))
    d_i = CategoricalLogPMF(logp=p, cats=[0, 1, 2, 3])

    samples = jnp.asarray(minp(d_i, p=0.4, n=_N))
    assert jnp.mean(samples) == 0
    samples = jnp.asarray(minp(d_i, p=0.3, n=_N))
    assert jnp.isclose(jnp.mean(samples), 0.5, atol=0.1)
    samples = jnp.asarray(minp(d_i, p=0.25, n=_N))
    assert jnp.isclose(jnp.mean(samples), 0.5, atol=0.1)
    samples = jnp.asarray(minp(d_i, p=0.2, n=_N))
    assert jnp.isclose(jnp.mean(samples), 1.3, atol=0.1)

    minp_samples = jnp.asarray(minp(d_i, p=0, n=_N))
    random_samples = jnp.asarray(random(d_i, n=_N))
    assert jnp.isclose(jnp.mean(minp_samples), jnp.mean(random_samples), atol=0.1)

    d_c = _make_char_unigram()
    assert minp(d_c, p=1) == greedy(d_c)

    with pytest.raises(ValueError, match="n must be at least 1"):
        minp(d_c, p=0.5, n=0)
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        minp(d_c, p=-0.1)
