"""Case 09 in JAX."""

import jax
import jax.numpy as jnp
import numpy as np

CENTERS = jnp.array([0.0, 0.5, 1.0, 1.5])


def ft_ao_buggy(Gv, centers=CENTERS):
    G = jnp.asarray(Gv, dtype=jnp.float32)
    phase = jnp.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


def ft_ao_fixed(Gv, centers=CENTERS):
    G = jnp.asarray(Gv, dtype=jnp.float32).reshape(-1, 3)
    phase = jnp.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


GRID = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])


# --------------------------------------------------------------------------- tests

def test_rank_polymorphism_is_a_feature_here_and_a_bug():
    """The same source traces to two different programs depending on the input rank."""
    assert ft_ao_buggy(GRID).shape == (3, 4)
    assert ft_ao_buggy(GRID[1]).shape == (4,)


def test_jit_specialises_on_each_rank_without_complaint():
    fn = jax.jit(ft_ao_buggy)
    assert fn(GRID).shape == (3, 4)
    assert fn(GRID[1]).shape == (4,)   # retraced, recompiled, still wrong


def test_fixed_is_rank_stable():
    assert ft_ao_fixed(GRID[1]).shape == (1, 4)
    assert np.allclose(np.asarray(ft_ao_fixed(GRID[1])), np.asarray(ft_ao_fixed(GRID[1].reshape(1, 3))))
