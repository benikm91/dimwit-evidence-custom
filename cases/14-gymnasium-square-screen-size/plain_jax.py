"""Case 14 in JAX. Nothing changes: the defect is a scalar reused for two concepts."""

import jax.numpy as jnp
import numpy as np


def resize(frame, out_h, out_w):
    in_h, in_w = frame.shape
    rows = jnp.clip(jnp.arange(out_h) * in_h // out_h, 0, in_h - 1)
    cols = jnp.clip(jnp.arange(out_w) * in_w // out_w, 0, in_w - 1)
    return frame[rows][:, cols]


def preprocess_buggy(frame, screen_size: int):
    return resize(frame, screen_size, screen_size)


def preprocess_fixed(frame, screen_size):
    return resize(frame, *screen_size)


FRAME = jnp.asarray(np.arange(210 * 160, dtype=np.float32).reshape(210, 160))


# --------------------------------------------------------------------------- tests

def test_both_are_valid_programs():
    assert preprocess_buggy(FRAME, 84).shape == (84, 84)
    assert preprocess_fixed(FRAME, (84, 64)).shape == (84, 64)


def test_shape_inference_confirms_a_square_output_which_is_exactly_the_bug():
    import jax
    shaped = jax.eval_shape(lambda f: preprocess_buggy(f, 84), FRAME)
    assert shaped.shape == (84, 84)


def test_there_is_no_mismatch_to_detect():
    """Every axis has the extent the program asked for. The program asked for the wrong one."""
    assert preprocess_buggy(FRAME, 84).shape[0] == preprocess_buggy(FRAME, 84).shape[1]
