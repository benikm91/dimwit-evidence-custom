"""Case 08 in JAX."""

import jax
import jax.numpy as jnp
import numpy as np

SAMPLE_DIMS = ("chain", "draw")


def to_point_list_buggy(data, dims, sample_dims=SAMPLE_DIMS):
    order = list(sample_dims) + [d for d in dims if d not in sample_dims]
    moved = jnp.transpose(data, [dims.index(d) for d in order])
    rest = data.shape[len(sample_dims):]
    return jnp.reshape(moved, (-1, *rest))


def to_point_list_fixed(data, dims, sample_dims=SAMPLE_DIMS):
    order = list(sample_dims) + [d for d in dims if d not in sample_dims]
    moved = jnp.transpose(data, [dims.index(d) for d in order])
    rest = moved.shape[len(sample_dims):]
    return jnp.reshape(moved, (-1, *rest))


# --------------------------------------------------------------------------- tests

def test_both_shapes_are_inferable_and_both_are_legal():
    data = jnp.arange(5 * 2 * 3, dtype=jnp.float32).reshape(5, 2, 3)
    dims = ("team", "draw", "chain")
    assert to_point_list_buggy(data, dims).shape == (10, 3)
    assert to_point_list_fixed(data, dims).shape == (6, 5)


def test_jit_compiles_the_buggy_version():
    """Static shapes throughout: XLA has no reason to object."""
    data = jnp.arange(5 * 2 * 3, dtype=jnp.float32).reshape(5, 2, 3)
    fn = jax.jit(lambda d: to_point_list_buggy(d, ("team", "draw", "chain")))
    assert fn(data).shape == (10, 3)


def test_reshape_checks_only_the_element_count():
    x = jnp.zeros((5, 2, 3))
    assert jnp.reshape(x, (10, 3)).size == jnp.reshape(x, (6, 5)).size == 30
    with np.testing.assert_raises(TypeError):
        jnp.reshape(x, (7, 5))
