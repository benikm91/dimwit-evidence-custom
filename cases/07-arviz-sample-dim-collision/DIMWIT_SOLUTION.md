# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Required: dimwit.tensor.Tensor1[Case07Buggy.Sample, dimwit.Float32]
where: ... (Case07Buggy.Chain |*| Case07Buggy.Draw, Case07Buggy.Sample) ...
    summariseUserSamples(flat.mean(Axis[Sample]))
                         ^^^^^^^^^^^^^^^^^^^^^^^
```

## Why it works — two distinct mechanisms

**1. The fused axis is not given a name; it is given a structure.**
`posterior.flatten((Axis[Chain], Axis[Draw]))` produces an axis labelled `Chain |*| Draw`.
There is no naming decision to make, so there is no name to collide. The upstream question
"what should we call the stacked dimension?" — answered first with `sample`, then with
`__sample__` — does not arise.

**2. Labels are types, so they live in the language's namespace.**
Two libraries can both define a label called `Sample`; they are `arviz.Sample` and
`myModel.Sample`, distinct types, and the compiler will not confuse them. String-keyed
systems have exactly one flat global namespace and no way to scope it.

## This is the case where DimWit beats the other named-axis systems

Worth stating explicitly, because most of the dossier compares DimWit against *positional*
APIs, where any named system would also do well. Here:

| system | axis identity | outcome |
|---|---|---|
| NumPy / JAX | position | missed (no names at all) |
| jaxtyping | size, named in a string | missed |
| xarray / ArviZ | runtime string, one flat namespace | **this is where the bug happened** |
| Haliax | runtime string, one flat namespace | same exposure |
| DimWit | Scala type, scoped, structured under `\|*\|` | compile-time |

## Honest limits

* Two DimWit labels with the same *short* name in the same scope are a Scala ambiguity, not
  a silent collision — the compiler complains. That is the desired behaviour, but it means
  the benefit is inherited from Scala's namespacing rather than invented by DimWit. Say so.
* `relabelTo` exists and can rename an axis deliberately. As always, DimWit checks that a
  value is used where its label says it belongs; it does not check that the label was
  chosen honestly.
