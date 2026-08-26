"""Case 13 in JAX."""

import jax
import jax.numpy as jnp
import numpy as np

WINDOW, HOP, N_MEL = 4, 2, 3
MEL_FILTERS = jnp.asarray(np.random.default_rng(13).uniform(size=(WINDOW, N_MEL)), dtype=jnp.float32)


def log_mel_single(audio):
    n = (audio.shape[0] - WINDOW) // HOP + 1
    frames = jnp.stack([jax.lax.dynamic_slice(audio, (i * HOP,), (WINDOW,)) for i in range(n)])
    return jnp.log(jnp.abs(frames @ MEL_FILTERS) + 1e-9)


AUDIO = jnp.arange(12, dtype=jnp.float32)
BATCH = jnp.stack([AUDIO, AUDIO + 100.0, AUDIO + 200.0, AUDIO + 300.0, AUDIO + 400.0])


# --------------------------------------------------------------------------- tests

def test_the_single_clip_path_is_correct():
    assert log_mel_single(AUDIO).shape == (5, N_MEL)


def test_the_batched_call_fails_at_runtime_here():
    """`dynamic_slice` is stricter than fancy slicing: this one does raise."""
    try:
        log_mel_single(BATCH)
    except Exception:
        return
    raise AssertionError("expected a rank error from dynamic_slice")


def test_vmap_is_the_fix_and_it_is_one_word():
    batched = jax.vmap(log_mel_single)
    assert batched(BATCH).shape == (5, 5, N_MEL)


def test_but_nothing_required_the_author_to_use_it():
    """The single-clip function's signature says nothing about how it should be lifted."""
    assert log_mel_single.__annotations__ == {}
