"""Case 10 under jaxtyping."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype

VOCAB, DIM = 7, 4
EMB = jnp.asarray(np.random.default_rng(3).normal(size=(VOCAB, DIM)), dtype=jnp.float32)
W_OUT = jnp.asarray(np.random.default_rng(4).normal(size=(DIM, VOCAB)), dtype=jnp.float32)


@jaxtyped(typechecker=beartype)
def model(idx: Int[Array, "batch seq"]) -> Float[Array, "batch seq vocab"]:
    x = EMB[idx]
    x = x - x.mean(axis=0, keepdims=True)
    return x @ W_OUT


IDX = jnp.array([1, 3, 5, 2])


# --------------------------------------------------------------------------- tests

def test_the_unbatched_call_is_rejected():
    """RUN-TIME detection: rank 1 where rank 2 was declared."""
    with pytest.raises(Exception):
        model(IDX)


def test_the_batched_call_is_accepted():
    assert model(IDX.reshape(1, -1)).shape == (1, 4, VOCAB)


def test_the_annotation_does_not_constrain_the_centring_step():
    """It catches the missing axis, not the operation that misused it.

    `x.mean(axis=0)` is still positional inside the body; if a later refactor moved the
    batch axis, the annotation would still pass and the centring would still be wrong.
    """
    out = model(IDX.reshape(1, -1))
    assert out.shape == (1, 4, VOCAB)
