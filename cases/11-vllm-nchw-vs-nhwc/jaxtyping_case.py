"""Case 11 under jaxtyping.

The annotation names the axes, which looks like exactly the right tool — and it is, right
up until height, width and channels all equal 3.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def per_channel_mean(frames: Float[Array, "frame height width channel"]) -> Float[Array, "channel"]:
    return frames.mean(axis=(0, 1, 2))


# --------------------------------------------------------------------------- tests

def test_square_rgb_frames_slip_through():
    """MISSED. [2, 3, 3, 3] satisfies "frame height width channel" under either layout."""
    nchw = jnp.zeros((2, 3, 3, 3))
    assert per_channel_mean(nchw).shape == (3,)


def test_a_non_square_frame_is_rejected():
    """RUN-TIME detection, but only once the extents stop coinciding."""
    nchw = jnp.zeros((2, 3, 4, 5))
    out = per_channel_mean(nchw)
    assert out.shape == (5,)   # the annotation is satisfied: "channel" simply binds to 5


def test_the_annotation_cannot_state_which_axis_is_the_channel():
    """`channel` is a size variable. It binds to whatever is in that position."""
    assert per_channel_mean(jnp.zeros((2, 3, 4, 5))).shape == (5,)
    assert per_channel_mean(jnp.zeros((2, 5, 4, 3))).shape == (3,)
