"""Case 07 in JAX.

JAX arrays have no axis names at all, so the collision cannot even be represented: the
names live in whatever bookkeeping the caller maintains alongside the array. That is a
stronger form of the same problem, not a weaker one.
"""

import jax.numpy as jnp
import numpy as np


def stack_chain_draw(data, dims, new_name):
    """`data` is a jnp array; `dims` is a Python tuple the caller has to keep in sync."""
    to_stack = ("chain", "draw")
    rest = [d for d in dims if d not in to_stack]
    order = [dims.index(d) for d in (*to_stack, *rest)]
    moved = jnp.transpose(data, order)
    n = int(np.prod([data.shape[dims.index(d)] for d in to_stack]))
    return jnp.reshape(moved, (n, *moved.shape[2:])), (new_name, *rest)


# --------------------------------------------------------------------------- tests

def test_jax_never_sees_the_names():
    data = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5)
    out, dims = stack_chain_draw(data, ("chain", "draw", "sample"), "sample")
    assert out.shape == (12, 5)
    assert dims == ("sample", "sample")  # duplicated, and jnp is entirely unaware


def test_the_array_is_perfectly_valid():
    data = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5)
    out, _ = stack_chain_draw(data, ("chain", "draw", "sample"), "sample")
    assert float(out.sum()) == float(data.sum())


def test_bookkeeping_can_drift_from_the_array_without_any_check():
    """The names and the array are two independent objects. Nothing keeps them consistent."""
    data = jnp.zeros((4, 3, 5))
    dims = ("chain", "draw", "sample", "extra")  # one name too many
    assert len(dims) != data.ndim  # no exception was raised anywhere
