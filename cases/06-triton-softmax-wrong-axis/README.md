# 06 — `tl.softmax` normalised along the wrong axis

**Source:** [triton-lang/triton#11406](https://github.com/triton-lang/triton/issues/11406)
· fix: [PR #11409](https://github.com/triton-lang/triton/pull/11409)
— *"[FRONTEND] Fix tl.softmax normalizing along the wrong axis"*

## The defect

`tl.softmax` reduced over the wrong axis of the tile. The output has the same shape and the
same dtype as the correct result; the numbers are normalised down the columns instead of
across the rows.

## Why this case is in the dossier

**Because DimWit does not catch it.** It is the honest counterweight to the other fourteen,
and the paper is stronger for including it.

Choosing the wrong reduction axis is not a *shape* error at all. `softmax` over axis 0 and
`softmax` over axis 1 of an `n x m` tile are both total functions from `n x m` to `n x m`.
No type discipline that describes shapes can prefer one over the other, and DimWit's cannot
either: `logits.vapply(Axis[Class])(softmax)` and `logits.vapply(Axis[Batch])(softmax)` are
both well-typed programs of the same type.

What DimWit changes is the *legibility* of the mistake, not its detectability — see
`DIMWIT_SOLUTION.md`. `Buggy.scala` compiles, and the harness reports it as `MISSED`.
That is the intended result, not a harness failure.
