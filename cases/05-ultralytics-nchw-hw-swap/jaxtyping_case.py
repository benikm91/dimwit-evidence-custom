"""Case 05 under jaxtyping.

Two versions, because the verdict depends on something easy to get wrong: jaxtyping binds
axis names only from *array* annotations. A `height: int` parameter does not bind the symbol
`height`, so a return annotation mentioning it constrains nothing.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, UInt8, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def to_model_input_int_args(
    image: UInt8[Array, "orig_h orig_w channel"],
    height: int,
    width: int,
) -> UInt8[Array, "1 channel height width"]:
    """The natural way to write it. `height` and `width` here are FREE variables."""
    resized = jnp.zeros((width, height, image.shape[2]), dtype=image.dtype)  # swapped
    return jnp.transpose(resized, (2, 0, 1))[None, ...]


@jaxtyped(typechecker=beartype)
def to_model_input_from_template_buggy(
    image: UInt8[Array, "orig_h orig_w channel"],
    template: UInt8[Array, "channel height width"],
) -> UInt8[Array, "1 channel height width"]:
    """The ONNX input tensor itself supplies the symbols, so now they are bound."""
    height, width = template.shape[1], template.shape[2]
    resized = jnp.zeros((width, height, image.shape[2]), dtype=image.dtype)  # swapped
    return jnp.transpose(resized, (2, 0, 1))[None, ...]


@jaxtyped(typechecker=beartype)
def to_model_input_from_template_fixed(
    image: UInt8[Array, "orig_h orig_w channel"],
    template: UInt8[Array, "channel height width"],
) -> UInt8[Array, "1 channel height width"]:
    height, width = template.shape[1], template.shape[2]
    resized = jnp.zeros((height, width, image.shape[2]), dtype=image.dtype)
    return jnp.transpose(resized, (2, 0, 1))[None, ...]


IMAGE = jnp.zeros((1080, 1920, 3), dtype=jnp.uint8)
TEMPLATE_RECT = jnp.zeros((3, 640, 480), dtype=jnp.uint8)
TEMPLATE_SQUARE = jnp.zeros((3, 640, 640), dtype=jnp.uint8)


# --------------------------------------------------------------------------- tests

def test_int_arguments_bind_nothing_so_the_swap_is_missed():
    """MISSED, even for a rectangular model. `height: int` does not bind `height`."""
    out = to_model_input_int_args(IMAGE, 640, 480)
    assert out.shape == (1, 3, 480, 640)   # transposed, and the annotation accepted it


def test_a_template_array_does_bind_them_and_the_swap_is_caught():
    """RUN-TIME detection — but only once the layout is carried by an array, not two ints."""
    with pytest.raises(Exception):
        to_model_input_from_template_buggy(IMAGE, TEMPLATE_RECT)


def test_even_with_a_template_a_square_model_hides_it():
    """MISSED for the default 640x640 export: `height` and `width` bind to the same number."""
    assert to_model_input_from_template_buggy(IMAGE, TEMPLATE_SQUARE).shape == (1, 3, 640, 640)


def test_the_fixed_version_is_accepted_for_both():
    assert to_model_input_from_template_fixed(IMAGE, TEMPLATE_RECT).shape == (1, 3, 640, 480)
    assert to_model_input_from_template_fixed(IMAGE, TEMPLATE_SQUARE).shape == (1, 3, 640, 640)
