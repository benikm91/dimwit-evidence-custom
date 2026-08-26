# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Found:    Tensor3[Case10Buggy.Batch, Case10Buggy.Pos, Case10Buggy.Embed, dimwit.Float32]
Required: Tensor2[Case10Buggy.Pos, Case10Buggy.Embed, dimwit.Float32]
    model(embeddings, wOut)
          ^^^^^^^^^^
```

## Why it works, and why the mechanism is bigger than the check

The compile error is the small part. The large part is that **`Fixed.scala`'s model has no
`Batch` in its signature at all.** A layer is a function on one example; batching is applied
from outside with `vmap`. Consequences:

* There is no batch axis to forget, so the most common shape fault in the literature has no
  place to occur.
* The op that caused the damage upstream is written `embeddings.mean(Axis[Pos])`, not
  `x.mean(axis=0)`. It reduces over positions because it says so, not because positions
  happen to be first. jaxtyping's residual gap — a refactor that moves the batch axis —
  does not exist here.
* `nextTokenLogits` slices `Axis[Pos]`, not `-1`. The lit-llama fix had to change
  `logits[-1]` to `logits[0, -1]`; the DimWit form does not mention position 0 of anything,
  so it does not need changing when a batch axis is added.

This is the case to use in the paper when arguing that DimWit's benefit is partly
*architectural* — vmap-first layers — and not only *type-theoretic*. Both halves should be
credited honestly: JAX offers `vmap` too, and a JAX author who used it consistently would
avoid this bug. DimWit's contribution is that the types make the vmap-first style checkable
rather than merely recommended.

## Honest limits

* A DimWit author can still write a layer with an explicit `Batch` axis in its signature.
  Nothing forbids it. The style is enforced by the library's own APIs (DeepWit's
  `MultiHeadAttention`, `CrossEntropy` and `Conv2DLayer` are all per-example), not by the
  type system.
* This is a rank error, the easiest class. jaxtyping catches it too, one phase later.
