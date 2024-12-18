import re
from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

from decoding.pmf import (
    LogPMF,
    cross_entropy,
    entropy,
    js_distance,
    js_divergence,
    kl_divergence,
    surprise,
)
from decoding.utils import getkey, logsoftmax


def _make_random_logpmf(size: int = 10) -> LogPMF[int]:
    logp = logsoftmax(jr.normal(getkey(), (size,)))
    return LogPMF(logp=logp, items=list(range(size)))


def test_categorical_logpmf() -> None:
    with pytest.raises(ValueError, match="LogProbs must be 1D"):
        LogPMF(logp=jnp.asarray([[0.0], [0.0]]), items=[0, 1])
    with pytest.raises(ValueError, match="LogProbs and Categories must match length"):
        LogPMF(logp=jnp.asarray([0.0, 0.0, 0.0]), items=[0, 1])
    with pytest.raises(ValueError, match="LogProbs must be proper distribution"):
        LogPMF(logp=jnp.asarray([0.0, 0.0]), items=[0, 1])

    c = [0, 1]
    for p in (jnp.asarray([0.5, 0.5]), jnp.asarray([0.0, 1.0])):
        d = LogPMF(logp=jnp.log(p), items=c)
        assert jnp.exp(d.logp).all() == p.all()
        assert d.items == c
        for _, cat in d:
            assert cat in c

    for bits in [0, 1, 10]:
        cats = list(range(2**bits))
        d = LogPMF.from_samples(samples=cats)
        nats = entropy(d)
        assert jnp.isclose(nats / jnp.log(2), bits)

    n = 1_000_000
    logits = n * jr.normal(getkey(), (n,))
    d = LogPMF.from_logits(logits=logits, items=list(range(n)))

    d1 = LogPMF.from_logits(logits=jnp.asarray([0.0, 0.0]), items=c)
    d2 = LogPMF.from_samples(samples=c * 10)
    assert c == d1.items == d2.items
    assert jnp.allclose(d1.logp, d2.logp)

    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'items'"):
        d.items = None  # type: ignore[reportAttributeAccessIssue]
    msg = re.escape(
        "Samples must be `Sequence[T]` or `Sequence[ScoredItem[T]]` and nonempty"
    )
    with pytest.raises(ValueError, match=msg):
        LogPMF.from_samples(samples=[])


def test_surprise() -> None:
    c = [0, 1]

    p = jnp.asarray([0.5, 0.5])
    d = LogPMF(logp=jnp.log(p), items=c)
    nats = surprise(d, 0)
    bits = nats / jnp.log(2)
    assert bits == 1.0

    p = jnp.asarray([0.0, 1.0])
    d = LogPMF(logp=jnp.log(p), items=c)
    assert surprise(d, 0) == jnp.inf
    assert surprise(d, 1) == 0.0

    p = jnp.asarray([1e-32, 1 - 1e-32])
    d = LogPMF(logp=jnp.log(p), items=c)
    assert 0 < surprise(d, 0) < jnp.inf

    with pytest.raises(ValueError, match="Category not in distribution"):
        surprise(d, 2)


def test_entropy() -> None:
    c = [0, 1]

    p = jnp.asarray([0.5, 0.5])
    d = LogPMF(logp=jnp.log(p), items=c)
    nats = entropy(d)
    bits = nats / jnp.log(2)
    assert bits == 1.0

    p = jnp.asarray([0.0, 1.0])
    d = LogPMF(logp=jnp.log(p), items=c)
    assert entropy(d) == 0.0

    p = jnp.asarray([1e-32, 1 - 1e-32])
    d = LogPMF(logp=jnp.log(p), items=c)
    assert entropy(d) > 0.0

    d = _make_random_logpmf()
    surps = jnp.asarray([surprise(d, i) for i in range(len(d.items))])
    h = entropy(d)
    assert jnp.isclose(h, jnp.sum(jnp.exp(d.logp) * surps))


def test_kl_divergence() -> None:
    logp = jnp.log(jnp.asarray([0.5, 0.5]))
    d_p = LogPMF(logp=logp, items=[0, 1])
    d_q = LogPMF(logp=logp, items=[1, 2])

    with pytest.raises(ValueError, match="Distributions must have the same categories"):
        kl_divergence(d_p, d_q)

    d_q = d_p
    kl = kl_divergence(d_p, d_q)
    assert kl == 0.0

    logp = jnp.log(jnp.asarray([1.0, 0.0]))
    d_q = LogPMF(logp=logp, items=[0, 1])
    kl_pq = kl_divergence(d_p, d_q)
    kl_qp = kl_divergence(d_q, d_p)
    assert kl_pq == jnp.inf
    assert kl_qp / jnp.log(2) == 1.0

    logp = jnp.log(jnp.asarray([0.6, 0.4]))
    d_q1 = LogPMF(logp=logp, items=[0, 1])
    logp = jnp.log(jnp.asarray([0.4, 0.6]))
    d_q2 = LogPMF(logp=logp, items=[0, 1])
    kl_pq1 = kl_divergence(d_p, d_q1)
    kl_pq2 = kl_divergence(d_p, d_q2)
    assert kl_pq1 == kl_pq2

    logp = jnp.log(jnp.asarray([0.7, 0.3]))
    q3 = LogPMF(logp=logp, items=[0, 1])
    kl_pq3 = kl_divergence(d_p, q3)
    assert kl_pq3 > kl_pq1


def test_cross_entropy() -> None:
    logp = jnp.log(jnp.asarray([0.5, 0.5]))
    d_p = LogPMF(logp=logp, items=[0, 1])
    d_q = d_p
    h_p = entropy(d_p)
    ce = cross_entropy(d_p, d_q)
    assert jnp.isclose(ce, h_p)

    logp = jnp.log(jnp.asarray([1.0, 0.0]))
    d_q = LogPMF(logp=logp, items=[0, 1])
    ce_pq = cross_entropy(d_p, d_q)
    ce_qp = cross_entropy(d_q, d_p)
    assert ce_pq == jnp.inf
    assert ce_qp / jnp.log(2) == 1.0

    d_p = _make_random_logpmf()
    d_q = _make_random_logpmf()
    h_p = entropy(d_p)
    h_q = entropy(d_q)
    kl_pq = kl_divergence(d_p, d_q)
    kl_qp = kl_divergence(d_q, d_p)
    ce_pq = cross_entropy(d_p, d_q)
    ce_qp = cross_entropy(d_q, d_p)
    assert jnp.isclose(kl_pq, ce_pq - h_p)
    assert jnp.isclose(kl_qp, ce_qp - h_q)


def test_js_divergence() -> None:
    logp = jnp.log(jnp.asarray([0.5, 0.5]))
    d_p = LogPMF(logp=logp, items=[0, 1])
    d_q = LogPMF(logp=logp, items=[1, 2])

    with pytest.raises(ValueError, match="Distributions must have the same categories"):
        js_divergence(d_p, d_q)

    d_q = d_p
    jsd = js_divergence(d_p, d_q)
    assert jsd == 0.0

    logp = jnp.log(jnp.asarray([1.0, 0.0]))
    d_q = LogPMF(logp=logp, items=[0, 1])
    jsd = js_divergence(d_p, d_q)
    assert 0.0 < jsd < jnp.inf

    d_p = _make_random_logpmf()
    d_q = _make_random_logpmf()
    jsd_pq = js_divergence(d_p, d_q)
    jsd_qp = js_divergence(d_q, d_p)
    assert jnp.isclose(jsd_pq, jsd_qp)


def test_js_distance() -> None:
    logp = jnp.log(jnp.asarray([0.5, 0.5]))
    d_p = LogPMF(logp=logp, items=[0, 1])
    d_q = d_p
    jsm = js_distance(d_p, d_q)
    assert jsm == 0.0

    logp = jnp.log(jnp.asarray([1.0, 0.0]))
    d_q = LogPMF(logp=logp, items=[0, 1])
    jsm = js_distance(d_p, d_q)
    assert 0.0 < jsm < jnp.inf

    d_p = _make_random_logpmf()
    d_q = _make_random_logpmf()
    jsm_pq = js_distance(d_p, d_q)
    jsm_qp = js_distance(d_q, d_p)
    assert jnp.isclose(jsm_pq, jsm_qp)


def test_jax_compare() -> None:
    d_p = _make_random_logpmf()
    d_q = _make_random_logpmf()

    assert jnp.isclose(
        entropy(d_p),
        jsp.special.entr(jnp.exp(d_p.logp)).sum(),
    )
    assert jnp.isclose(
        kl_divergence(d_p, d_q),
        jsp.special.kl_div(jnp.exp(d_p.logp), jnp.exp(d_q.logp)).sum(),
    )


def test_category_types() -> None:
    from collections import deque

    logp = logsoftmax(jr.normal(getkey(), (2,)))
    for objs in [[0, 1], ["a", "b"], [int, str]]:
        for typ in [list, tuple, deque]:
            cats = typ(objs)
            d = LogPMF(logp=logp, items=cats)
            assert surprise(d, cats[0]) >= 0.0
