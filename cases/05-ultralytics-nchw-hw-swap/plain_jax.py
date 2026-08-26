"""Case 05 in JAX: transposing the spatial axes of an image is a legal array operation."""

import jax.numpy as jnp
import numpy as np


def preprocess_buggy(image_hwc, onnx_input_shape):
    """Reads (N, C, H, W) as width-then-height, so it resizes into a transposed buffer."""
    width, height = onnx_input_shape[2], onnx_input_shape[3]
    resized = jnp.zeros((height, width, image_hwc.shape[2]), dtype=image_hwc.dtype)
    return jnp.transpose(resized, (2, 0, 1))[None, ...]  # to NCHW


def preprocess_fixed(image_hwc, onnx_input_shape):
    height, width = onnx_input_shape[2], onnx_input_shape[3]
    resized = jnp.zeros((height, width, image_hwc.shape[2]), dtype=image_hwc.dtype)
    return jnp.transpose(resized, (2, 0, 1))[None, ...]


# --------------------------------------------------------------------------- tests

def test_both_produce_a_valid_nchw_batch():
    img = jnp.zeros((1080, 1920, 3), dtype=jnp.uint8)
    b = preprocess_buggy(img, (1, 3, 640, 480))
    f = preprocess_fixed(img, (1, 3, 640, 480))
    assert b.ndim == f.ndim == 4


def test_the_spatial_axes_are_transposed():
    """Erroneous behaviour: (1, 3, 480, 640) where the model declared (1, 3, 640, 480)."""
    img = jnp.zeros((1080, 1920, 3), dtype=jnp.uint8)
    assert preprocess_buggy(img, (1, 3, 640, 480)).shape == (1, 3, 480, 640)
    assert preprocess_fixed(img, (1, 3, 640, 480)).shape == (1, 3, 640, 480)


def test_a_square_model_input_hides_it_completely():
    img = jnp.zeros((1080, 1920, 3), dtype=jnp.uint8)
    assert preprocess_buggy(img, (1, 3, 640, 640)).shape == preprocess_fixed(img, (1, 3, 640, 640)).shape
