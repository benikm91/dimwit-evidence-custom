"""Case 13 — a single-waveform feature extractor called on a batch (NumPy).

Original: openai/whisper#839.

The framing step slices the FIRST axis and the mel projection contracts the SECOND, so both
survive an extra leading axis. That is why the batched call runs instead of crashing.
"""

import numpy as np
import pytest

WINDOW, HOP, N_MEL = 4, 2, 3
RNG = np.random.default_rng(13)
MEL_FILTERS = RNG.uniform(size=(WINDOW, N_MEL))


def log_mel_single(audio):
    """Written for a 1-D waveform. `audio[i : i + WINDOW]` slices time."""
    n = (len(audio) - WINDOW) // HOP + 1
    frames = np.stack([audio[i * HOP : i * HOP + WINDOW] for i in range(n)])
    mel = np.tensordot(frames, MEL_FILTERS, axes=([1], [0]))
    return np.log(np.abs(mel) + 1e-9)


def log_mel_batched_buggy(batch):
    """The call that was being made: a [batch, samples] array into the single version."""
    return log_mel_single(batch)


def log_mel_batched_fixed(batch):
    """After PR #839: the single-clip function, applied per clip."""
    return np.stack([log_mel_single(clip) for clip in batch])


AUDIO = np.arange(12, dtype=float)
BATCH = np.stack([AUDIO + 100.0 * k for k in range(5)])   # 5 clips of 12 samples


# --------------------------------------------------------------------------- tests

def test_the_single_clip_path_is_correct():
    assert log_mel_single(AUDIO).shape == (5, N_MEL)


def test_the_batched_call_runs_without_an_exception():
    """Erroneous behaviour: no crash, and a 3-D result that looks like [frame, ?, mel]."""
    out = log_mel_batched_buggy(BATCH)
    assert out.ndim == 3
    assert np.isfinite(out).all()


def test_it_framed_across_clips_instead_of_across_time():
    """`len(audio)` was the batch size, so there is one "frame" made of four whole clips."""
    assert log_mel_batched_buggy(BATCH).shape == (1, 12, N_MEL)
    assert log_mel_batched_fixed(BATCH).shape == (5, 5, N_MEL)


def test_the_frame_axis_now_counts_clips_not_time_steps():
    """The middle axis is 12 — the sample count — which no correct output would ever have."""
    buggy = log_mel_batched_buggy(BATCH)
    assert buggy.shape[1] == BATCH.shape[1]


def test_a_batch_too_short_to_frame_crashes_instead():
    """With fewer clips than the window there are no frames at all, and stack raises."""
    with pytest.raises(ValueError, match="at least one array"):
        log_mel_batched_buggy(BATCH[:1])
