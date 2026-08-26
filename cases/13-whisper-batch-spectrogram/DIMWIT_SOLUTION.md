# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:21:18
Found:    (batch : Tensor2[Case13Fixed.Batch, Case13Fixed.Sample, Float32])
Required: Tensor1[Case13Fixed.Sample, Float32]
    logMelSingle(batch, filters)
                 ^^^^^
```

Two things are checked here, and the second is the one the other tools miss. The rank
mismatch at the boundary is caught — jaxtyping catches that too, at run time. But the actual
damage upstream was *inside* the body: the framing step sliced the first axis, so with a
batch present it framed across clips instead of across time. In DimWit that step is
`slice(Axis[Sample].at(...))`, which names the axis it cuts, so adding a leading axis later
cannot silently repoint it — and `stack(frames, Axis[Frame])` gives the frame axis its own
identity rather than leaving it as "whatever axis 0 is now".
