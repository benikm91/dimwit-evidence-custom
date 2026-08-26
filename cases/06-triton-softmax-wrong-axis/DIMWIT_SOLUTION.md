# DimWit — verdict

**Category: `missed`**

Verified: `scala-cli compile Buggy.scala` **succeeds**. `Buggy.scala` even runs, printing
the wrong row sums. This is a real negative result and belongs in the paper as one.

## Why DimWit cannot catch it

`vapply(Axis[L])(softmax)` has type
`Tensor2[Batch, Class, Float32] => Tensor2[Batch, Class, Float32]` for `L = Class` and the
same type for `L = Batch`. Choosing the reduction axis of a shape-preserving reduction is a
choice between two programs of identical type. No shape-indexed type system can prefer one,
because the distinction is not in the shapes.

The same holds for `mean`, `std`, `cumsum`, `argsort`, layer normalisation, and every other
keep-shape reduction — a large family of real defects that DimWit does not address.

## What DimWit does change

Not detection. **Legibility**, and it is a narrower claim than it looks:

| | plain | DimWit |
|---|---|---|
| the mistake reads as | `dim=0` | `Axis[Batch]` |
| to review it you must | recall the layout convention at this point in the program | read the identifier |
| renaming/reordering axes upstream | silently changes which axis is reduced | cannot change it |

That last row is the only part with teeth. In NumPy or PyTorch, inserting a batch axis
upstream silently repoints every positional `dim=` downstream; in DimWit `Axis[Class]` still
names `Class`. So DimWit removes the *action at a distance*, while leaving the local choice
unchecked.

## How to report this in the paper

State it as a limitation, in the same sentence as the mechanism, e.g.:

> DimWit's types constrain which axes a value *has*, not which axis an operation *should*
> use. Shape-preserving reductions — softmax, mean, layer norm — are therefore outside its
> reach; the axis becomes a name rather than an index, which helps review but is not a
> guarantee.

A table with fifteen wins and no losses is not evidence, it is advertising. This row and
case 14 are the two that make the rest believable.
