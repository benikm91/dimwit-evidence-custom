"""Case 06 under jaxtyping. There is nothing for an annotation to constrain."""

import jax.numpy as jnp
import numpy as np
from jax.nn import softmax
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def softmax_buggy(logits: Float[Array, "batch cls"]) -> Float[Array, "batch cls"]:
    return softmax(logits, axis=0)


@jaxtyped(typechecker=beartype)
def softmax_fixed(logits: Float[Array, "batch cls"]) -> Float[Array, "batch cls"]:
    return softmax(logits, axis=-1)


# --------------------------------------------------------------------------- tests

def test_both_satisfy_the_annotation_even_when_the_sizes_differ():
    """MISSED unconditionally. Unlike cases 01 and 05, distinct sizes do not help.

    A reduction that keeps its shape is shape-preserving whichever axis it reduces, so no
    size information distinguishes the two programs.
    """
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)  # batch=3, cls=4, deliberately unequal
    assert softmax_buggy(x).shape == (3, 4)
    assert softmax_fixed(x).shape == (3, 4)


def test_results_differ():
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    assert not np.allclose(np.asarray(softmax_buggy(x)), np.asarray(softmax_fixed(x)))
