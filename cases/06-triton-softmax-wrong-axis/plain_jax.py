"""Case 06 in JAX."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import softmax


def softmax_buggy(logits):
    return softmax(logits, axis=0)


def softmax_fixed(logits):
    return softmax(logits, axis=-1)


# --------------------------------------------------------------------------- tests

def test_both_are_the_same_function_type():
    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    assert jax.eval_shape(softmax_buggy, x) == jax.eval_shape(softmax_fixed, x)


def test_results_differ():
    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    assert not np.allclose(np.asarray(softmax_buggy(x)), np.asarray(softmax_fixed(x)))


def test_a_cross_entropy_loss_built_on_the_wrong_axis_still_trains():
    """The most dangerous property: the wrong objective is still differentiable and finite."""
    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    labels = jnp.array([0, 1, 2, 3])

    def loss(z):
        p = softmax_buggy(z)
        return -jnp.mean(jnp.log(p[jnp.arange(4), labels] + 1e-9))

    g = jax.grad(loss)(x)
    assert g.shape == (4, 4)
    assert bool(jnp.isfinite(g).all())
