# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Found:    dimwit.tensor.Tensor1[Case01Buggy.Source, dimwit.Bool]
Required: dimwit.tensor.Tensor1[Case01Buggy.Target, dimwit.Bool]
    maskTargetPositions(scores, sourcePaddingMask(sourceTokens))
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

## Why it works

Two independent mechanisms, and it is worth separating them in the paper because only the
first is a *detection*:

1. **The mistake is not expressible in the idiomatic program.** A padding mask over key
   positions has type `Tensor1[Source, Bool]`. Broadcasting it against
   `Tensor2[Target, Source, Float32]` can align it with `Source` and nothing else — there
   is no `None`-insertion whose position the author could get wrong. `Fixed.scala` is
   therefore not "the fixed version" so much as the only version that can be written.

2. **The intent, once written down, is checked.** `Buggy.scala` reconstructs the mistake
   faithfully: the author believed the mask lines up with the query axis and wrote a
   helper taking `Tensor1[Target, Bool]`. Feeding it the source mask is rejected, because
   `Source` and `Target` are distinct types even though both axes have extent 3.

## What carries the guarantee

Axis **identity**, not axis **extent**. That distinction is the whole argument: the two
sequence axes in self-attention have the same length, so no amount of size reasoning can
tell them apart, whereas `Source` and `Target` are as different as `String` and `Int`.

## Honest limits

* Getting the mask's *label* wrong at construction — building it from the target token ids
  instead of the source ones — would still be accepted. DimWit checks that a value is used
  where its label says it belongs; it cannot check that the label was assigned truthfully.
* Nothing here depends on the extents matching. If the two sequences had different lengths,
  a runtime check in JAX or jaxtyping would also have caught it. The claim is about the
  case where they do not differ, which is the case that actually shipped.
