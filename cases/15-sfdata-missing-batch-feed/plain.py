"""Case 15 — feeding an example without its batch axis (NumPy).

Original: SFData program s55237206 (tensfa/tensfa), from StackOverflow 55237206.
Reconstructed without TensorFlow: `feed` reproduces the placeholder check that raised.
"""

import numpy as np
import pytest

IMAGE = (100, 100, 3)
N_CLASSES = 2
FLAT = IMAGE[0] * IMAGE[1] * IMAGE[2]

RNG = np.random.default_rng(15)
W = RNG.normal(size=(FLAT, N_CLASSES)) * 0.01
B = np.zeros(N_CLASSES)


def feed(placeholder_shape, value):
    """What tf.Session.run does with a feed_dict entry."""
    if tuple(value.shape) != tuple(placeholder_shape):
        raise ValueError(
            f"Cannot feed value of shape {tuple(value.shape)} for Tensor Placeholder:0, "
            f"which has shape {tuple(placeholder_shape)}"
        )
    return value


def train_step_buggy(x_all, y_all, i):
    """As posted: the i-th example is fed straight in."""
    x = feed((1, *IMAGE), x_all[i])
    y = feed((None, N_CLASSES), y_all[i])
    return x, y


def train_step_fixed(x_all, y_all, i):
    """The dataset's ground-truth patch."""
    x = feed((1, *IMAGE), np.expand_dims(x_all[i], 0))
    y = feed((1, N_CLASSES), np.expand_dims(y_all[i], 0))
    return x, y


def _data(n=3):
    x = RNG.random((n, *IMAGE)).astype(np.float32)
    y = np.eye(N_CLASSES)[RNG.integers(0, N_CLASSES, n)]
    return x, y


# --------------------------------------------------------------------------- tests

def test_the_buggy_feed_raises_the_upstream_error():
    """RUN-TIME detection, and this is the exact message from the StackOverflow post."""
    x, y = _data()
    with pytest.raises(ValueError, match=r"Cannot feed value of shape \(100, 100, 3\)"):
        train_step_buggy(x, y, 0)


def test_the_patched_feed_is_accepted():
    x, y = _data()
    fed_x, fed_y = train_step_fixed(x, y, 0)
    assert fed_x.shape == (1, *IMAGE)
    assert fed_y.shape == (1, N_CLASSES)


def test_the_reshape_path_would_have_been_silent():
    """The model's first op is `tf.reshape(X, [-1, 30000])`, which accepts both ranks.

    Had the placeholder been declared `(None, 100, 100, 3)` — the usual thing — nothing
    would have raised, because reshape only checks the element count.
    """
    x, _ = _data()
    assert x[0].reshape(-1, FLAT).shape == (1, FLAT)
    assert x[0][None].reshape(-1, FLAT).shape == (1, FLAT)


def test_the_label_path_is_silently_wrong_rather_than_loud():
    """A one-hot label of shape (2,) against logits of shape (1, 2) broadcasts fine."""
    logits = np.zeros((1, N_CLASSES))
    label = np.eye(N_CLASSES)[1]           # shape (2,)
    assert (label * logits).shape == (1, N_CLASSES)   # no error; the batch axis is invented
