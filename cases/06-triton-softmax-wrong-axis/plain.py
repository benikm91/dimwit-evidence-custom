"""Case 06 — softmax normalised along the wrong axis (NumPy).

Original: triton-lang/triton#11406, fixed by PR #11409.
"""

import numpy as np


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_buggy(logits):
    """Reduces over the batch axis."""
    return softmax(logits, axis=0)


def softmax_fixed(logits):
    """Reduces over the class axis, which is what a classifier means by softmax."""
    return softmax(logits, axis=-1)


def _logits():
    """Deliberately square: 4 examples, 4 classes. The shipping shape of many tiles."""
    return np.arange(16, dtype=np.float64).reshape(4, 4)


# --------------------------------------------------------------------------- tests

def test_shapes_and_dtypes_are_identical():
    x = _logits()
    assert softmax_buggy(x).shape == softmax_fixed(x).shape == (4, 4)
    assert softmax_buggy(x).dtype == softmax_fixed(x).dtype


def test_buggy_rows_are_not_distributions():
    """Erroneous behaviour: the rows do not sum to 1, the columns do."""
    p = softmax_buggy(_logits())
    assert not np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p.sum(axis=0), 1.0)


def test_fixed_rows_are_distributions():
    p = softmax_fixed(_logits())
    assert np.allclose(p.sum(axis=1), 1.0)


def test_every_value_is_still_a_plausible_probability():
    """Nothing downstream can flag it: the output is in [0, 1] and sums to 1 somewhere."""
    p = softmax_buggy(_logits())
    assert (p >= 0).all() and (p <= 1).all()


def test_a_single_row_batch_makes_every_class_equally_certain():
    """A batch of one is the worst case: normalising over it turns every logit into 1.0."""
    x = np.array([[1.0, 5.0, 2.0]])
    p = softmax_buggy(x)
    assert np.allclose(p, 1.0)                 # every class "certain", no error raised
    assert np.allclose(softmax_fixed(x).sum(), 1.0)
