# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with

```
Found:    (decoded : Case11Buggy.Nchw)
Required: Case11Buggy.Nhwc
    perChannelMean(decoded)
                   ^^^^^^^
```

The message is short because the layouts were given names — `type Nchw` and `type Nhwc` —
which is itself part of the point: the convention that lived in a docstring upstream is a
declaration here.

## Why it works

`Tensor4[Frame, Channel, Height, Width, Float32]` and
`Tensor4[Frame, Height, Width, Channel, Float32]` are different types **regardless of their
extents**. The square-RGB coincidence that hid the bug from every other tool in this
comparison is irrelevant: `Channel` in position 1 is not `Height` in position 1.

The reduction is also stated rather than implied:
`frames.mean((Axis[Frame], Axis[Height], Axis[Width]))` names the axes it collapses, so
`perChannelMean` cannot quietly reduce the channel axis even if the layout changed.

## Where this case sits in the argument

It is the cleanest "layout is a type" example in the dossier, and the one where the size
coincidence is most complete — under NumPy the two tensors are *literally the same shape*.
Use it in the paper wherever the claim is that DimWit distinguishes things that shape
inference provably cannot.

## Honest limits

* The guarantee stops at the FFI boundary. Whatever the decoder hands over is a raw buffer;
  someone has to assert its layout when lifting it into a `Tensor4[...]`, and that assertion
  is unchecked. DimWit moves the trust boundary to one line instead of spreading it over the
  pipeline — a real improvement, but not the elimination of trust.
* Two decoders returning the *same* DimWit type but different physical layouts would still
  be confused. The type records what the programmer declared, not what the bytes contain.
