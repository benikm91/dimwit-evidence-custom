# DimWit — verdict

**Category: `missed`**

Verified: `scala-cli compile Buggy.scala` **succeeds**, and the file runs, printing that a
210x160 frame became 84x84.

## Why DimWit cannot catch it

`Shape(Axis[Height] -> s, Axis[Width] -> s)` must be well-typed for every `s`, because square
tensors are ordinary and useful. So a single integer used for two named axes cannot be an
error. `AxisExtent[Height]` and `AxisExtent[Width]` are distinct types — that is what saved
cases 04 and 05 — but *constructing* both from the same `Int` is exactly what the constructor
`Axis[L] -> n` is for.

Generalising: DimWit's types constrain **how values may be combined**, not **which values a
program should have chosen**. Case 06 is the same limit seen from the other side — there the
unconstrained choice was an axis, here it is an extent.

## What DimWit changes

Only legibility, and the honest version of that claim is modest:

```scala
// DimWit                                        // Python
Shape(Axis[Height] -> screenSize,                (screen_size, screen_size)
      Axis[Width]  -> screenSize)
```

Both are one number used twice. The DimWit form names the two concepts at the point of
reuse, so a reader sees the square assumption being made rather than having to know that the
tuple is `(h, w)`. That helps review. It is not detection, and the paper should not dress it
up as such.

## Why this row belongs in the paper

Together with case 06 it delimits the claim. The dossier's earlier estimate for this case was
"partly prevented"; building it out showed that is too generous, and the correct verdict is
`missed` with an API-design mitigation. Reporting that correction is worth more to a
sceptical reader than a fifteenth green cell.
