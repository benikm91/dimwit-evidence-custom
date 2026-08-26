# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Found:    (ys : dimwit.tensor.Tensor1[Case02Buggy.Batch, dimwit.Float32])
Required: dimwit.tensor.Tensor[(Case02Buggy.Batch, Case02Buggy.Out), DType.Float32]
    val residuals = preds - ys
                            ^^
```

## Why it works

Not because of names — because of the **operator split**. In DimWit:

* `-` requires both operands to have the *identical* label tuple. No broadcasting, ever.
* `-!` broadcasts, and has to be typed by the author.

The buggy program uses `-`, so the missing `Out` axis is a type error. Had the author
written `preds -! ys` it would compile and reproduce the PyTorch behaviour exactly — but
that `!` is a decision recorded in the source, visible in review and in a diff.

## This is the row where named axes alone are not enough

Worth stating explicitly in the paper. Haliax also has named axes, and it broadcasts by
name *implicitly*: subtracting a `{batch}` array from a `{batch, out}` array aligns on
`batch` and silently produces `{batch, out}`, exactly as PyTorch does. Naming the axes does
not help when the mistake is the broadcast itself.

So this case isolates a design decision that is independent of the names-as-types thesis:
**implicit broadcasting is opt-in**. It is the one place in the dossier where DimWit is the
only tool of the four that rejects the program.

## Honest limits

* `Fixed.scala` avoids the problem twice over: the per-example model returns `Tensor0`, so
  the `Out` axis of width 1 never exists. That is idiomatic DimWit, but it means the
  comparison is partly between *styles* and not only between *type systems* — say so.
* `preds -! ys` compiles. DimWit makes the broadcast explicit; it does not make it
  impossible, and an author who reaches for `!` to silence a compiler error reintroduces
  the bug. The guarantee is "no silent broadcast", not "no wrong broadcast".
