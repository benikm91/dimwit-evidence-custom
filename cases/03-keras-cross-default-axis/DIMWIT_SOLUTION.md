# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Found:    dimwit.tensor.Tensor1[Case03Buggy.Sample, dimwit.Float32]
Required: dimwit.tensor.Tensor1[Case03Buggy.Component, dimwit.Float32]
    zipvmap(Axis[Component])(a, b)((x: Tensor1[Sample, ...], y: ...) => cross3(x, y))
```

## Why it works

`cross3` is typed `Tensor1[Component, Float32] => Tensor1[Component, Float32]`. The
component axis is part of the function's identity, not something recovered from the shape
at call time. Two consequences:

1. **There is no default to get wrong.** DimWit has no notion of "the axis that happens to
   have extent 3", because axes are not addressed by extent at all. The entire mechanism
   that produced the upstream bug is absent.
2. **Batching over the wrong axis is a type error.** `zipvmap(Axis[Component])` hands
   `cross3` the `Sample` slices, and `Sample` is not `Component` — even though both have
   extent 3 in the failing example.

## Why this case matters most for the paper

It separates the two candidate explanations for DimWit's benefit. It is *not* that DimWit
knows more about sizes — the sizes are identical in both programs and would be identical
under any size-based checker. It is that `Sample` and `Component` are **different types**,
so the compiler can reject a program that is perfectly well-shaped.

Every other tool in the comparison is size-based, and every one of them misses this.

## Honest limits

* The guarantee comes from `cross3`'s signature. A DimWit author who wrote
  `cross3(t: Tensor1[L, Float32])` generically, for any label `L`, would get no protection —
  the types would unify with `Sample` happily. Naming the axis in the API is the discipline;
  the type system enforces the discipline once adopted, it does not supply it.
* `Fixed.scala` still fails at run time if the `Component` axis does not have extent 3
  (the `slice(Axis[Component].at(2))` is a runtime bounds check). Extents remain runtime
  values in DimWit.
