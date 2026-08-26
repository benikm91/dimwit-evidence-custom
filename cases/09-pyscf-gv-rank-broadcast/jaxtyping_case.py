"""Case 09 under jaxtyping — a clean win for annotations, at run time."""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

CENTERS = jnp.array([0.0, 0.5, 1.0, 1.5])


@jaxtyped(typechecker=beartype)
def ft_ao(Gv: Float[Array, "gpoint 3"]) -> Float[Array, "gpoint center"]:
    phase = jnp.exp(-0.5 * (Gv**2).sum(axis=-1))
    return phase[..., None] * CENTERS


GRID = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])


# --------------------------------------------------------------------------- tests

def test_a_rank_one_argument_is_rejected():
    """RUN-TIME detection: `Float[Array, "gpoint 3"]` requires rank 2."""
    with pytest.raises(Exception):
        ft_ao(GRID[1])


def test_the_reshaped_call_is_accepted():
    assert ft_ao(GRID[1].reshape(1, 3)).shape == (1, 4)


def test_rank_is_the_one_thing_annotations_always_pin_down():
    """Unlike sizes, rank cannot coincide by accident, so this class is reliably caught."""
    assert ft_ao(GRID).shape == (3, 4)
    with pytest.raises(Exception):
        ft_ao(jnp.zeros((2, 2, 3)))
