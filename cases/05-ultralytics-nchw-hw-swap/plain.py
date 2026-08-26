"""Case 05 — NCHW height/width read in the wrong order (NumPy).

Original: ultralytics/ultralytics#23126, fixed by PR #23402.
"""

import numpy as np


def read_input_size_buggy(onnx_input_shape):
    """As shipped: `input_shape` is (N, C, H, W) and the two are swapped."""
    return {"width": onnx_input_shape[2], "height": onnx_input_shape[3]}


def read_input_size_fixed(onnx_input_shape):
    """After PR #23402."""
    return {"height": onnx_input_shape[2], "width": onnx_input_shape[3]}


def letterbox_shape(size):
    """The (H, W, C) array the preprocessing allocates."""
    return (size["height"], size["width"], 3)


ONNX_SHAPE_RECT = (1, 3, 640, 480)   # H = 640, W = 480
ONNX_SHAPE_SQUARE = (1, 3, 640, 640)


# --------------------------------------------------------------------------- tests

def test_the_two_readers_disagree_on_a_rectangular_model_input():
    buggy = read_input_size_buggy(ONNX_SHAPE_RECT)
    fixed = read_input_size_fixed(ONNX_SHAPE_RECT)
    assert buggy == {"width": 640, "height": 480}
    assert fixed == {"height": 640, "width": 480}


def test_the_bug_is_invisible_on_a_square_model_input():
    """YOLO's default imgsz is 640x640, so the defect never showed in the common path."""
    assert read_input_size_buggy(ONNX_SHAPE_SQUARE) == read_input_size_fixed(ONNX_SHAPE_SQUARE)


def test_the_letterbox_buffer_is_transposed():
    """Erroneous behaviour: the preprocessing allocates (480, 640, 3) instead of (640, 480, 3)."""
    assert letterbox_shape(read_input_size_buggy(ONNX_SHAPE_RECT)) == (480, 640, 3)
    assert letterbox_shape(read_input_size_fixed(ONNX_SHAPE_RECT)) == (640, 480, 3)


def test_numpy_allocates_the_transposed_buffer_without_complaint():
    """No error: (480, 640, 3) is a perfectly good array."""
    buf = np.zeros(letterbox_shape(read_input_size_buggy(ONNX_SHAPE_RECT)), dtype=np.uint8)
    assert buf.size == 480 * 640 * 3


def test_the_aspect_ratio_of_the_model_input_is_inverted():
    """The concrete consequence: boxes are later rescaled with the axes swapped."""
    buggy = read_input_size_buggy(ONNX_SHAPE_RECT)
    fixed = read_input_size_fixed(ONNX_SHAPE_RECT)
    assert buggy["width"] / buggy["height"] == 640 / 480
    assert fixed["width"] / fixed["height"] == 480 / 640
