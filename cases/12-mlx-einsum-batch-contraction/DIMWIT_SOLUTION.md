# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with DimWit's own diagnostic

```
Axis[Case12Buggy.Row] not found in Tensor[(Case12Buggy.Inner, Case12Buggy.Col)].
    a.dot(Axis[Row])(b)
          ^^^^^^^^^
```

## Why it works

`dot(Axis[L])(other)` requires `L` to be present in **both** operands' label tuples. The
mistake is therefore not "contract the wrong axis" but "contract an axis that one operand
does not have", which is a type error rather than a numerical one.

Two things follow:

* The contraction is **symmetric and named**. In `einsum` the shared letter is a positional
  convention inside a string; here it is a type that both operands must carry.
* The **batch axis is not part of the contraction at all**. `Fixed.scala` writes an ordinary
  matrix product and lifts it with `vmap(Axis[Batch])`. The upstream bug was specifically
  about batch dimensions not being broadcast in a batched tensordot — an entire category of
  question that does not arise when batching is a combinator rather than a subscript letter.

## Honest limits

* Contracting a genuinely shared but wrong axis — say two operands that both carry `Inner`
  and `Col`, and the author contracts `Col` — would type-check. DimWit checks that the axis
  *exists in both*, not that it is the *intended* one. The upstream defect happens to be of
  the first kind; a neighbouring one would not be.
* As always, extents are unchecked: two tensors both labelled `Inner` with extents 3 and 4
  fail at run time.
