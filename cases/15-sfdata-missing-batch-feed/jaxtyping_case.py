"""Case 15 under jaxtyping."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

IMAGE = (100, 100, 3)
N_CLASSES = 2
FLAT = IMAGE[0] * IMAGE[1] * IMAGE[2]
W = jnp.asarray(np.random.default_rng(15).normal(size=(FLAT, N_CLASSES)) * 0.01, dtype=jnp.float32)


@jaxtyped(typechecker=beartype)
def logits(x: Float[Array, "batch h w c"]) -> Float[Array, "batch cls"]:
    return jnp.reshape(x, (-1, FLAT)) @ W


@jaxtyped(typechecker=beartype)
def loss(
    logits_: Float[Array, "batch cls"],
    labels: Float[Array, "batch cls"],
) -> Float[Array, ""]:
    return -jnp.mean(jnp.sum(labels * jnp.log(jnp.abs(logits_) + 1e-9), axis=-1))


X = jnp.asarray(np.random.default_rng(15).random((3, *IMAGE)), dtype=jnp.float32)


# --------------------------------------------------------------------------- tests

def test_the_unbatched_feature_is_rejected():
    """RUN-TIME detection: rank 3 where rank 4 was declared."""
    with pytest.raises(Exception):
        logits(X[0])


def test_the_unbatched_label_is_rejected_too():
    """The label path, which broadcast silently in plain JAX, is now caught."""
    with pytest.raises(Exception):
        loss(logits(X[:1]), jnp.eye(N_CLASSES)[1])


def test_the_batched_calls_are_accepted():
    assert logits(X[:1]).shape == (1, N_CLASSES)
    assert float(loss(logits(X[:1]), jnp.eye(N_CLASSES)[:1])) == pytest.approx(
        float(loss(logits(X[:1]), jnp.eye(N_CLASSES)[:1]))
    )
