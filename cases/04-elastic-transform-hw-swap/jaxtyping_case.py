"""Case 04 under jaxtyping.

The tensors are annotated correctly and the defect is entirely in the two scalar divisors,
which no array annotation reaches.
"""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def normalise_buggy(
    dx: Float[Array, "height width"],
    dy: Float[Array, "height width"],
    alpha_x: float,
    alpha_y: float,
    height: int,
    width: int,
) -> tuple[Float[Array, "height width"], Float[Array, "height width"]]:
    return dx * alpha_x / height, dy * alpha_y / width  # swapped


@jaxtyped(typechecker=beartype)
def normalise_fixed(
    dx: Float[Array, "height width"],
    dy: Float[Array, "height width"],
    alpha_x: float,
    alpha_y: float,
    height: int,
    width: int,
) -> tuple[Float[Array, "height width"], Float[Array, "height width"]]:
    return dx * alpha_x / width, dy * alpha_y / height


# --------------------------------------------------------------------------- tests

def test_the_annotation_is_satisfied_by_the_buggy_version():
    """MISSED. Every array has the shape it claims; only the divisors are swapped."""
    dx = jnp.ones((100, 800))
    a, b = normalise_buggy(dx, dx, 50.0, 50.0, 100, 800)
    assert a.shape == b.shape == (100, 800)


def test_int_arguments_carry_no_axis_identity():
    """`height: int` and `width: int` are the same type. Swapping them type-checks."""
    dx = jnp.ones((100, 800))
    a, _ = normalise_buggy(dx, dx, 50.0, 50.0, 800, 100)  # arguments swapped: still fine
    assert a.shape == (100, 800)


def test_results_differ():
    dx = jnp.ones((100, 800))
    a, _ = normalise_buggy(dx, dx, 50.0, 50.0, 100, 800)
    c, _ = normalise_fixed(dx, dx, 50.0, 50.0, 100, 800)
    assert not np.allclose(np.asarray(a), np.asarray(c))
