"""
Miscellaneous helper functions.
"""

import secrets

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from decoding.types import FVX, KEY


def getkey() -> KEY:
    """
    Get a random key for use in JAX functions.

    Returns:
        A random key.

    Examples:
        ```python
        import jax.random as jr
        from decoding.utils import getkey

        key = getkey()
        x = jr.normal(key, (10,))
        assert x.shape == (10,)
        ```

    """
    return jr.PRNGKey(secrets.randbelow(2**32))


def logsoftmax(x: FVX, *, t: float = 1.0) -> FVX:
    """
    Compute the log-softmax of a vector.

    Args:
        x: The input vector.
        t: The temperature of the softmax.

    Returns:
        The log-softmax of the input vector.

    Examples:
        ```python
        import jax.numpy as jnp
        import jax.nn as jnn
        from decoding.utils import logsoftmax

        x = jnp.array([1.0, 2.0, 3.0])
        logp = logsoftmax(x)
        assert jnn.logsumexp(logp) == 0.0
        ```

    """
    if t == 0:
        logp = jnp.where(jnp.arange(x.size) == jnp.argmax(x), 0.0, -jnp.inf)
    elif t == float("inf"):
        logp = jnp.full_like(x, -jnp.log(x.size))
    else:
        logp = jnn.log_softmax(x / t)
    return logp
