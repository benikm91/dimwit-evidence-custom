"""Case 14 — a single `screen_size` forcing square observations (NumPy).

Original: Farama-Foundation/Gymnasium#1312.
"""

import numpy as np

ATARI_FRAME = (210, 160)


def _resize(frame, out_hw):
    """Nearest-neighbour resize, enough to show what the aspect ratio does."""
    out_h, out_w = out_hw
    in_h, in_w = frame.shape
    rows = (np.arange(out_h) * in_h // out_h).clip(0, in_h - 1)
    cols = (np.arange(out_w) * in_w // out_w).clip(0, in_w - 1)
    return frame[np.ix_(rows, cols)]


def preprocess_buggy(frame, screen_size: int):
    """As shipped: one integer, used for both axes."""
    return _resize(frame, (screen_size, screen_size))


def preprocess_fixed(frame, screen_size):
    """After PR #1312: `screen_size` is (height, width)."""
    height, width = screen_size
    return _resize(frame, (height, width))


def _frame():
    return np.arange(ATARI_FRAME[0] * ATARI_FRAME[1], dtype=float).reshape(ATARI_FRAME)


# --------------------------------------------------------------------------- tests

def test_the_buggy_api_cannot_express_a_non_square_target():
    """The defect is an API limitation: there is no argument for a rectangle."""
    out = preprocess_buggy(_frame(), 84)
    assert out.shape == (84, 84)


def test_the_aspect_ratio_is_destroyed():
    """210:160 is 1.31:1; the output is 1:1. Every observation is vertically squashed."""
    frame = _frame()
    assert frame.shape[0] / frame.shape[1] > 1.3
    assert preprocess_buggy(frame, 84).shape[0] / preprocess_buggy(frame, 84).shape[1] == 1.0


def test_the_fixed_api_preserves_it_when_asked():
    out = preprocess_fixed(_frame(), (84, 64))
    assert out.shape == (84, 64)
    assert abs(out.shape[0] / out.shape[1] - 210 / 160) < 0.02


def test_nothing_here_is_a_shape_error():
    """Both outputs are valid frames. The mistake is semantic, not structural."""
    assert preprocess_buggy(_frame(), 84).ndim == preprocess_fixed(_frame(), (84, 64)).ndim == 2


def test_a_square_input_would_have_made_the_two_identical():
    square = np.zeros((160, 160))
    assert preprocess_buggy(square, 84).shape == preprocess_fixed(square, (84, 84)).shape
