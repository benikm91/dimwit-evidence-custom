"""Case 13 in JAX — the same program as `plain.py`.

The one difference in outcome is that `jax.lax.dynamic_slice` is stricter than NumPy's
fancy slicing, so the batched call raises here instead of silently framing across clips.
"""

import jax
import jax.numpy as jnp
import numpy as np

WINDOW, HOP, N_MEL = 4, 2, 3
MEL_FILTERS = jnp.asarray(np.random.default_rng(13).uniform(size=(WINDOW, N_MEL)), dtype=jnp.float32)


def log_mel_single(audio):
    """Written for a 1-D waveform. The framing slices time."""
    n = (audio.shape[0] - WINDOW) // HOP + 1
    frames = jnp.stack([jax.lax.dynamic_slice(audio, (i * HOP,), (WINDOW,)) for i in range(n)])
    return jnp.log(jnp.abs(frames @ MEL_FILTERS) + 1e-9)


def log_mel_batched_buggy(batch):
    """The call that was being made: a [batch, samples] array into the single version."""
    return log_mel_single(batch)


def log_mel_batched_fixed(batch):
    """After PR #839: the single-clip function, applied per clip."""
    return jax.vmap(log_mel_single)(batch)


AUDIO = jnp.arange(12, dtype=jnp.float32)
BATCH = jnp.stack([AUDIO + 100.0 * k for k in range(5)])   # 5 clips of 12 samples


# --------------------------------------------------------------------------- tests

def test_the_single_clip_path_is_correct():
    assert log_mel_single(AUDIO).shape == (5, N_MEL)


def test_the_batched_call_fails_at_runtime_here():
    """RUN-TIME detection: `dynamic_slice` will not slice a rank-2 array with one index."""
    with np.testing.assert_raises(Exception):
        log_mel_batched_buggy(BATCH)


def test_the_fixed_version_frames_each_clip():
    assert log_mel_batched_fixed(BATCH).shape == (5, 5, N_MEL)


def test_but_nothing_required_the_author_to_lift_it():
    """The single-clip function's signature says nothing about how it should be lifted."""
    assert log_mel_single.__annotations__ == {}
