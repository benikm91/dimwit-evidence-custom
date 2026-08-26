"""Case 12 in JAX."""

import jax
import jax.numpy as jnp
import numpy as np

A = jnp.asarray(np.random.default_rng(12).normal(size=(3, 3, 3)), dtype=jnp.float32)
B = jnp.asarray(np.random.default_rng(13).normal(size=(3, 4)), dtype=jnp.float32)


def buggy(a, b):
    return jnp.tensordot(a, b, axes=([1], [0]))


def fixed(a, b):
    return jnp.tensordot(a, b, axes=([2], [0]))


# --------------------------------------------------------------------------- tests

def test_both_shapes_are_inferable():
    assert jax.eval_shape(buggy, A, B).shape == jax.eval_shape(fixed, A, B).shape == (3, 3, 4)


def test_the_results_differ():
    assert not np.allclose(np.asarray(buggy(A, B)), np.asarray(fixed(A, B)))


def test_vmap_is_the_thing_that_removes_the_choice():
    """Lifting a plain matrix-vector product over the batch leaves no axis to pick."""
    per_example = jax.vmap(lambda m: m @ B)
    assert np.allclose(np.asarray(per_example(A)), np.asarray(fixed(A, B)))
