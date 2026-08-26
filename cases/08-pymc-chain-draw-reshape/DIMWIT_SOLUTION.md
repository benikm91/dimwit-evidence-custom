# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Cannot merge axes TM in shape S. Ensure all axes exist..
I found:
  AxesMerger.bridge[(Case08Buggy.Draw, Case08Buggy.Chain, Case08Buggy.Team),
                    UnwrapAxes[(Axis[Draw], Axis[Chain])],
                    (Case08Buggy.Chain |*| Case08Buggy.Draw, Case08Buggy.Team)]
    ordered.flatten((Axis[Draw], Axis[Chain]))
```

The message is an implicit-resolution failure rather than a plain `Found/Required`, but the
two shapes are both in it: the program produced `Draw |*| Chain` where `Chain |*| Draw` was
required.

## Why it works

**DimWit has no `reshape`.** The operation that caused the upstream bug — "lay these
elements out in this rectangle of integers" — does not exist in the API. What exists is:

* `flatten((Axis[Chain], Axis[Draw]))`, whose result type records *which* axes were fused
  and *in which order*, as `Chain |*| Draw`; and
* `unflatten(Axis[Chain |*| Draw], Shape(chains, draws))`, which demands a `Shape` whose
  labels reconstruct that composite.

So the transpose and the flatten cannot disagree: both are written in terms of the same
labels, and the label of the fused axis is derived from them rather than asserted
separately. There is no second, positional description of the same operation to fall out of
sync.

Note the ordering guarantee specifically: `Draw |*| Chain` and `Chain |*| Draw` are
different types even though the fused axis has extent 6 either way. Elements are interleaved
differently, and that difference is in the type.

## Honest limits

* The compiler error here is noisier than in cases 01–05, because the check is carried by an
  implicit (`AxesMerger`) rather than by a direct type ascription. If the paper shows
  compiler output, show this one too rather than only the clean ones — reviewers will ask.
* `unflatten` still takes the *extents* at run time. Passing `Axis[Chain] -> 2, Axis[Draw] -> 3`
  where the true split is 3 and 2 is a run-time failure, not a compile-time one. Order is
  typed; sizes are not.
