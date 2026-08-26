"""Case 14 under jaxtyping."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


def _resize(frame, out_h, out_w):
    in_h, in_w = frame.shape
    rows = jnp.clip(jnp.arange(out_h) * in_h // out_h, 0, in_h - 1)
    cols = jnp.clip(jnp.arange(out_w) * in_w // out_w, 0, in_w - 1)
    return frame[rows][:, cols]


@jaxtyped(typechecker=beartype)
def preprocess_buggy(
    frame: Float[Array, "in_h in_w"], screen_size: int
) -> Float[Array, "out_h out_w"]:
    return _resize(frame, screen_size, screen_size)


@jaxtyped(typechecker=beartype)
def preprocess_fixed(
    frame: Float[Array, "in_h in_w"], height: int, width: int
) -> Float[Array, "out_h out_w"]:
    return _resize(frame, height, width)


FRAME = jnp.asarray(np.arange(210 * 160, dtype=np.float32).reshape(210, 160))


# --------------------------------------------------------------------------- tests

def test_the_annotation_is_satisfied_by_the_square_output():
    """MISSED. `out_h` and `out_w` are free variables; binding both to 84 is fine."""
    assert preprocess_buggy(FRAME, 84).shape == (84, 84)


def test_an_annotation_could_only_forbid_squares_which_is_wrong():
    """There is no annotation that expresses "these two axes should usually differ"."""
    assert preprocess_fixed(FRAME, 84, 84).shape == (84, 84)
    assert preprocess_fixed(FRAME, 84, 64).shape == (84, 64)
