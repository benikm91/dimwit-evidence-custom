"""Case 13 under jaxtyping."""

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
    n = (audio.shape[0] - WINDOW) // HOP + 1
    frames = jnp.stack([jax.lax.dynamic_slice(audio, (i * HOP,), (WINDOW,)) for i in range(n)])
    return jnp.log(jnp.abs(frames @ MEL_FILTERS) + 1e-9)


AUDIO = jnp.arange(12, dtype=jnp.float32)
BATCH = jnp.stack([AUDIO, AUDIO + 100.0, AUDIO + 200.0, AUDIO + 300.0, AUDIO + 400.0])


# --------------------------------------------------------------------------- tests

def test_the_batched_call_is_rejected():
    """RUN-TIME detection: `Float[Array, "samples"]` is rank 1."""
    with pytest.raises(Exception):
        log_mel_single(BATCH)


def test_the_single_call_is_accepted():
    assert log_mel_single(AUDIO).shape == (5, N_MEL)


def test_vmap_composes_with_the_annotation():
    """The annotated function still lifts, and the lifted version is the correct one."""
    assert jax.vmap(log_mel_single)(BATCH).shape == (5, 5, N_MEL)
