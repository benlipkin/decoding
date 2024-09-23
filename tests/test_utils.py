import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from decoding.types import FS
from decoding.utils import getkey, logsoftmax


def test_getkeygen() -> None:
    key1 = getkey()
    key2 = getkey()
    assert not jnp.all(key1 == key2)


def test_logsoftmax() -> None:
    x = jnp.asarray([0.0, 1.0])
    logp = logsoftmax(x, t=0.0)
    assert jnp.allclose(logp, jnp.asarray([-jnp.inf, 0.0]))
    assert jnp.allclose(logp.max(), logsoftmax(x, t=1e-32).max(), atol=1e-6)

    logp = logsoftmax(x, t=float("inf"))
    assert jnp.allclose(logp, jnp.asarray([-jnp.log(2), -jnp.log(2)]))
    assert jnp.allclose(logp, logsoftmax(x, t=1e32), atol=1e-6)

    logp1 = logsoftmax(x, t=2.0)
    logp2 = logsoftmax(x, t=0.2)
    assert logp1[0] > logp2[0]
    assert logp1[1] < logp2[1]

    size = 10
    x = jr.normal(getkey(), (size,))
    ts = 10 ** jnp.linspace(-4, 4, 9)

    def test(t: FS) -> None:
        logp = logsoftmax(x, t=t.item())
        assert jnp.isclose(jnn.logsumexp(logp), 0.0, atol=1e-6)

    list(map(test, ts))
