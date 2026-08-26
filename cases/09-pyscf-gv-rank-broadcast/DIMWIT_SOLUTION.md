# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Found:    (g : dimwit.tensor.Tensor1[Case09Buggy.Component, dimwit.Float32])
Required: dimwit.tensor.Tensor2[Case09Buggy.GPoint, Case09Buggy.Component, dimwit.Float32]
    ftAo(g, centers)
         ^
```

## Why it works

Rank is part of a DimWit tensor's type — `Tensor1[...]` and `Tensor2[...]` are aliases for
`Tensor[T, V]` with tuples of different length — and DimWit never prepends a missing axis.
There is no rule that turns a `Tensor1[Component]` into a `Tensor2[GPoint, Component]`, so
the call simply does not type-check.

Note that the axis is not just *present*, it is *named*: the error says the missing axis is
`GPoint`. Compare NumPy's silence, and compare jaxtyping's message, which reports a rank
mismatch without saying which concept the missing axis stood for.

## The structural point

`Fixed.scala` writes the physics for one G-vector and obtains the grid version with `vmap`.
That is the same move as case 10's `Batch` and case 13's audio batching, and it is the
general answer to this whole family: **a function that operates on one thing should be typed
for one thing, and lifted.** When batching is a combinator rather than a convention, there
is no leading axis to forget.

## Honest limits

* This is a rank error, which is the easiest kind to catch. jaxtyping catches it too, one
  phase later. The DimWit-specific gain here is the *timing* and the *name in the message*,
  not the detection itself — do not oversell this row.
* Nothing checks that the `Component` axis has extent 3. `Fixed.scala` would happily accept
  a 2-dimensional reciprocal space and fail, or silently misbehave, at run time.
