# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails on the label path with

```
Found:    (batchY : Tensor[Case15Buggy.Class *: EmptyTuple, DType.Float32])
Required: Tensor2[Case15Buggy.Batch, Case15Buggy.Class, dimwit.Float32]
    batchLoss(batchX, batchY, w, b)
                      ^^^^^^
```

(the feature path fails for the same reason; the compiler reports the first error it reaches)

## Why it works

`images.slice(Axis[Batch].at(i))` *removes* the `Batch` axis from the type — the result is a
`Tensor3[Height, Width, Channel, Float32]`, and there is no rule that will put a batch axis
back. Slicing and feeding are therefore connected in a way they are not in NumPy, where
`x[i]` produces an array that remembers nothing about what was dropped.

`Fixed.scala` shows the structural answer: the loss is written for one image and one label,
and `zipvmap(Axis[Batch])` lifts it over the batch. The question the SFData program got
wrong — "does this thing I am feeding have a batch axis?" — is answered by the type of the
function being called, not by a comment.

## Connecting the dossier to the corpus

This case is the representative of SFData's largest class: 65.8% of its 146 crashing tensor
shape faults are feature-input or label-output incompatibilities. The four rank cases here
(09, 10, 13, 15) are all instances, and DimWit rejects all four at compile time.

That is the natural bridge from fifteen hand-picked defects to a dataset-scale claim, and it
is the experiment worth running next: classify all 146 SFData patches by the DimWit
mechanism that would have prevented them, and report the fraction.

## Honest limits

* This is a rank error at a call boundary, the easiest class. jaxtyping catches it too, and
  TensorFlow caught half of it. The DimWit gain is phase, not novelty.
* `Fixed.scala` uses `flatten` to reach `Height |*| Width |*| Channel`, which fixes the
  flattening order in the type — a nice side effect, but it also means the weight matrix's
  label has to be written out in full, which is verbose. Worth mentioning as an ergonomic
  cost if the paper discusses usability.
