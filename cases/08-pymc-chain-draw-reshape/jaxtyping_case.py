"""Case 08 under jaxtyping.

An annotation on the *output* would catch this — if the author had written one, and if the
two rectangles happen to differ. Both conditions are worth spelling out.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def to_point_list_buggy(
    data: Float[Array, "team draw chain"],
) -> Float[Array, "chain_draw team"]:
    moved = jnp.transpose(data, (2, 1, 0))          # -> (chain, draw, team)
    rest = data.shape[2:]                            # (chain,) : the pre-transpose tail
    return jnp.reshape(moved, (-1, *rest))


@jaxtyped(typechecker=beartype)
def to_point_list_fixed(
    data: Float[Array, "team draw chain"],
) -> Float[Array, "chain_draw team"]:
    moved = jnp.transpose(data, (2, 1, 0))
    rest = moved.shape[2:]
    return jnp.reshape(moved, (-1, *rest))


# --------------------------------------------------------------------------- tests

def test_the_buggy_output_violates_the_annotation():
    """RUN-TIME detection: `team` is bound to 5 by the input, but the output has 3 columns."""
    data = jnp.arange(5 * 2 * 3, dtype=jnp.float32).reshape(5, 2, 3)
    with pytest.raises(Exception):
        to_point_list_buggy(data)


def test_the_fixed_output_satisfies_it():
    data = jnp.arange(5 * 2 * 3, dtype=jnp.float32).reshape(5, 2, 3)
    assert to_point_list_fixed(data).shape == (6, 5)


def test_it_is_missed_when_the_two_rectangles_coincide():
    """team == chain, so (chain*draw, chain) and (chain*draw, team) are the same shape."""
    data = jnp.arange(3 * 2 * 3, dtype=jnp.float32).reshape(3, 2, 3)  # team = chain = 3
    assert to_point_list_buggy(data).shape == to_point_list_fixed(data).shape == (6, 3)
