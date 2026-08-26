"""Case 07 under jaxtyping.

jaxtyping's axis names look like the thing that would help here. They are not: the names
bind to *sizes*, and two axes whose sizes agree are interchangeable regardless of name.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def summarise_user_samples(x: Float[Array, "sample param"]) -> Float[Array, "param"]:
    """Meant to average over the user's `sample` dimension."""
    return jnp.mean(x, axis=0)


# --------------------------------------------------------------------------- tests

def test_a_stacked_chain_draw_axis_is_accepted_as_a_sample_axis():
    """MISSED. 4 chains x 3 draws = 12; a user `sample` axis of 12 is indistinguishable."""
    stacked = jnp.zeros((12, 5))  # (chain*draw, param)
    assert summarise_user_samples(stacked).shape == (5,)


def test_it_is_accepted_even_when_the_sizes_were_never_meant_to_match():
    """The annotation constrains rank and size agreement, never provenance."""
    anything_by_five = jnp.zeros((7, 5))
    assert summarise_user_samples(anything_by_five).shape == (5,)


def test_only_a_rank_error_is_caught():
    import pytest
    with pytest.raises(Exception):
        summarise_user_samples(jnp.zeros((4, 3, 5)))
