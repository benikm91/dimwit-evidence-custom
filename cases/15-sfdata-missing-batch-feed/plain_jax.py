"""Case 15 in JAX. Without an explicit placeholder declaration there is nothing to check."""

import jax
import jax.numpy as jnp
import numpy as np

IMAGE = (100, 100, 3)
N_CLASSES = 2
FLAT = IMAGE[0] * IMAGE[1] * IMAGE[2]
W = jnp.asarray(np.random.default_rng(15).normal(size=(FLAT, N_CLASSES)) * 0.01, dtype=jnp.float32)


def logits(x):
    """`tf.matmul(tf.reshape(X, [-1, input]), W)` — rank-polymorphic by construction."""
    return jnp.reshape(x, (-1, FLAT)) @ W


X = jnp.asarray(np.random.default_rng(15).random((3, *IMAGE)), dtype=jnp.float32)


# --------------------------------------------------------------------------- tests

def test_a_single_example_and_a_batch_of_one_are_indistinguishable_after_reshape():
    """MISSED. reshape(-1, FLAT) erases the difference the placeholder would have caught."""
    assert logits(X[0]).shape == (1, N_CLASSES)
    assert logits(X[0][None]).shape == (1, N_CLASSES)
    assert np.allclose(np.asarray(logits(X[0])), np.asarray(logits(X[0][None])))


def test_a_real_batch_also_works_so_nothing_flags_the_intent():
    assert logits(X).shape == (3, N_CLASSES)


def test_the_label_side_broadcasts_a_missing_batch_axis_into_existence():
    label = jnp.eye(N_CLASSES)[1]
    assert (label * logits(X[0])).shape == (1, N_CLASSES)


def test_jit_specialises_on_each_rank():
    fn = jax.jit(logits)
    assert fn(X[0]).shape == (1, N_CLASSES)
    assert fn(X).shape == (3, N_CLASSES)
