# 03 — `ops.cross` picked the first length-3 axis instead of the last

**Source:** [keras-team/keras#23219](https://github.com/keras-team/keras/pull/23219)
— *"Fix torch backend cross() using wrong default axis"*

## The defect

Keras' torch backend forwarded `axis=None` straight to `torch.cross`, whose documented
default is **the first dimension of size 3**. NumPy, JAX and the TensorFlow backend all use
the **last** axis. On any input where an earlier axis also happens to have length 3 the
backends silently disagree:

```python
a = np.arange(9.).reshape(3, 3)
b = np.arange(9.)[::-1].reshape(3, 3)
keras.ops.cross(a, b)   # torch: axis 0.  numpy: axis 2.  Same shape, different numbers.
```

The fix defaults `axis` to `-1` before the call, and adds a `(2, 3, 3)` regression test.

## Why it is interesting

This is the purest example of the general problem: **an API that identifies an axis by its
extent is ambiguous exactly when two axes share an extent.** `3` is the most collision-prone
extent in numerical code — RGB, spatial coordinates, quaternion-adjacent layouts, and the
cross product itself all use it.

Every checker in this comparison that reasons about sizes is structurally unable to help
here, because the sizes are identical in the buggy and the correct program.
