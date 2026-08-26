# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Found:    (batch : Tensor2[Case13Buggy.Batch, Case13Buggy.Sample, dimwit.Float32])
Required: Tensor1[Case13Buggy.Sample, dimwit.Float32]
    logMel(batch, filters)
           ^^^^^
```

## Why it works

The rank check is the visible part; the more interesting part is inside the body.

`Fixed.scala` frames the audio with `audio.slice(Axis[Sample].at(i * Hop until ...))`. The
axis being sliced is **named**. Compare the Python: `audio[i : i + WINDOW]` slices axis 0,
whatever axis 0 happens to be. That is precisely the operation that went wrong upstream, and
it is the operation that no boundary annotation constrains — jaxtyping guarantees the
argument is rank 1, but if a batch axis were later added, `audio[i : i + WINDOW]` would
silently start slicing it while the annotation was dutifully updated.

The framed result is also relabelled explicitly: a window of samples becomes
`Tensor1[Window, Float32]` via `relabelTo`, so the subsequent contraction against the mel
filter bank is checked against `Window` rather than against "whatever is left".

## Honest limits

* `relabelTo` is an unchecked assertion — it is how the author says "these samples are now a
  window". Reasonable, but it is a place where the type system takes the programmer's word.
* Nothing verifies that the window length matches the filter bank's `Window` extent; that is
  a run-time check, as extents always are.
* This is a rank error at the boundary, so jaxtyping catches it too. The DimWit-specific
  gain is the named slicing inside the body, which is the part that would survive a refactor.
