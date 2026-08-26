"""Case 04 in JAX. The divisors are Python ints; JAX never sees an axis at all."""

import jax
import jax.numpy as jnp
import numpy as np


def normalise_buggy(dx, dy, alpha, size):
    """size = (height, width)."""
    return dx * alpha[0] / size[0], dy * alpha[1] / size[1]


def normalise_fixed(dx, dy, alpha, size):
    return dx * alpha[0] / size[1], dy * alpha[1] / size[0]


# --------------------------------------------------------------------------- tests

def test_both_trace_and_jit_cleanly():
    dx = jnp.ones((100, 800))
    dy = jnp.ones((100, 800))
    for fn in (normalise_buggy, normalise_fixed):
        a, b = jax.jit(fn, static_argnums=(2, 3))(dx, dy, (50.0, 50.0), (100, 800))
        assert a.shape == b.shape == (100, 800)


def test_results_differ_on_non_square_input():
    dx = jnp.ones((100, 800))
    dy = jnp.ones((100, 800))
    ba, bb = normalise_buggy(dx, dy, (50.0, 50.0), (100, 800))
    fa, fb = normalise_fixed(dx, dy, (50.0, 50.0), (100, 800))
    assert not np.allclose(np.asarray(ba), np.asarray(fa))


def test_shape_inference_has_nothing_to_say():
    """The divisor is a scalar. Shape inference cannot distinguish 1/100 from 1/800."""
    dx = jnp.ones((100, 800))
    shaped = jax.eval_shape(lambda d: d * (50.0 / 100), dx)
    assert shaped.shape == (100, 800)
