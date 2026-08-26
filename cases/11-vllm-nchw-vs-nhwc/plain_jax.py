"""Case 11 in JAX."""

import jax
import jax.numpy as jnp
import numpy as np


def per_channel_mean_nhwc(frames):
    return frames.mean(axis=(0, 1, 2))


def to_nhwc(frames_nchw):
    return jnp.transpose(frames_nchw, (0, 2, 3, 1))


FRAMES_NCHW = jnp.asarray(np.random.default_rng(11).normal(size=(2, 3, 3, 3)), dtype=jnp.float32)


# --------------------------------------------------------------------------- tests

def test_shape_inference_accepts_the_wrong_layout():
    assert jax.eval_shape(per_channel_mean_nhwc, FRAMES_NCHW).shape == (3,)


def test_the_statistics_differ():
    wrong = per_channel_mean_nhwc(FRAMES_NCHW)
    right = per_channel_mean_nhwc(to_nhwc(FRAMES_NCHW))
    assert not np.allclose(np.asarray(wrong), np.asarray(right))


def test_a_normalisation_built_on_them_still_produces_a_valid_image():
    """The downstream tensor has the right shape and a plausible range: nothing to flag."""
    normalised = FRAMES_NCHW - per_channel_mean_nhwc(FRAMES_NCHW)
    assert normalised.shape == (2, 3, 3, 3)
    assert bool(jnp.isfinite(normalised).all())
