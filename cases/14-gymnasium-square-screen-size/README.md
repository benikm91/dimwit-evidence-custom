# 14 — `AtariPreprocessing` assumed the screen was square

**Source:** [Farama-Foundation/Gymnasium#1312](https://github.com/Farama-Foundation/Gymnasium/pull/1312)
— *"Allow `AtariPreprocessing` non-square observations"*

## The defect

`AtariPreprocessing(screen_size=84)` took a single integer and resized every observation to
`84 x 84`. Atari frames are `210 x 160`, so the aspect ratio was silently destroyed; and a
user who wanted a non-square target had no way to ask for one. The PR widens the parameter
to accept a tuple.

## Why this case is in the dossier

**Because DimWit does not catch it either.** It is the second honest miss, alongside case 06,
and the two of them are what make the other thirteen believable.

The defect is not a mismatch between two shapes — it is *one number used for two different
concepts*. `Shape(Axis[Height] -> s, Axis[Width] -> s)` is a perfectly well-typed expression
in DimWit for any `s`, and it must be, because square tensors are legitimate. `Buggy.scala`
compiles, and the harness reports it as `MISSED`. That is the intended result.

What DimWit changes is that the assumption is *written down*: the two axes are named at the
point where the same integer is used for both, so a reviewer sees `Axis[Height] -> size,
Axis[Width] -> size` rather than `(size, size)`. See `DIMWIT_SOLUTION.md`.
