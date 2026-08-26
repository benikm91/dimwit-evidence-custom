# 09 — a single G-vector passed as `(3,)` was silently broadcast

**Source:** [pyscf/pyscf#2961](https://github.com/pyscf/pyscf/issues/2961)
· fix: [PR #3340](https://github.com/pyscf/pyscf/pull/3340)
— *"Fix ft_aopair/ft_ao silent broadcasting error for 1D Gv"*

## The defect

`ft_ao` and `ft_aopair` compute Fourier transforms of atomic-orbital products over a grid of
reciprocal-space vectors `Gv`, expected as shape `(N, 3)`. Passing a *single* vector as
`(3,)` — the obvious thing to do when you want one point — did not raise. NumPy broadcast
the rank-1 array, and the routine returned an array of the wrong rank with wrong values.

The fix is a single line in each function:

```python
Gv = numpy.asarray(Gv, dtype=numpy.double).reshape(-1, 3)
```

and the regression test asserts `dat.shape == ref.shape` for `g` versus `g.reshape(1, 3)`.

## Why it is interesting

* **Not deep learning.** This is a quantum-chemistry code, which is useful for the paper's
  claim that the fault class is not specific to neural networks.
* **A rank error, not an extent error.** The distinction between "one vector" and "a list of
  one vector" is exactly the distinction between `Tensor1[Component]` and
  `Tensor2[GPoint, Component]`, and rank *is* in DimWit's types.
* Same family as case 02 (`[n, 1]` vs `[n]`) and case 10 (the missing batch axis): NumPy's
  rule that a missing leading axis is prepended silently is doing the damage in all three.
