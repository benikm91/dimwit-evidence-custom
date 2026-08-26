"""Case 13 under jaxtyping — `plain_jax.py` with the axis names written into annotations.

`Float[Array, "samples"]` pins the rank, so the batched call is rejected at run time. What
the annotation does not reach is the framing inside the body: `dynamic_slice(audio, (i,), ...)`
is still positional, and nothing says it slices time rather than clips.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

WINDOW, HOP, N_MEL = 4, 2, 3
MEL_FILTERS = jnp.asarray(np.random.default_rng(13).uniform(size=(WINDOW, N_MEL)), dtype=jnp.float32)


@jaxtyped(typechecker=beartype)
def log_mel_single(audio: Float[Array, "samples"]) -> Float[Array, "frame mel"]:
    """Written for a 1-D waveform. The framing slices time."""
    n = (audio.shape[0] - WINDOW) // HOP + 1
    frames = jnp.stack([jax.lax.dynamic_slice(audio, (i * HOP,), (WINDOW,)) for i in range(n)])
    return jnp.log(jnp.abs(frames @ MEL_FILTERS) + 1e-9)


@jaxtyped(typechecker=beartype)
def log_mel_batched_buggy(batch: Float[Array, "batch samples"]) -> Float[Array, "frame mel"]:
    """The call that was being made: a [batch, samples] array into the single version."""
    return log_mel_single(batch)


@jaxtyped(typechecker=beartype)
def log_mel_batched_fixed(batch: Float[Array, "batch samples"]) -> Float[Array, "batch frame mel"]:
    """After PR #839: the single-clip function, applied per clip."""
    return jax.vmap(log_mel_single)(batch)


AUDIO = jnp.arange(12, dtype=jnp.float32)
BATCH = jnp.stack([AUDIO + 100.0 * k for k in range(5)])


# --------------------------------------------------------------------------- tests

def test_the_batched_call_is_rejected():
    """RUN-TIME detection: `Float[Array, "samples"]` is rank 1."""
    with pytest.raises(Exception):
        log_mel_batched_buggy(BATCH)


def test_the_single_call_is_accepted():
    assert log_mel_single(AUDIO).shape == (5, N_MEL)


def test_vmap_composes_with_the_annotation():
    """The annotated function still lifts, and the lifted version is the correct one."""
    assert log_mel_batched_fixed(BATCH).shape == (5, 5, N_MEL)
