# DimWit — verdict

**Category: `compile-time detection`**

Verified: `scala-cli compile Buggy.scala` fails with two errors, one per swapped argument:

```
Found:    dimwit.tensor.Axis[Case05Buggy.Width]
Required: dimwit.tensor.Axis[Case05Buggy.Height]
    val boxed = letterbox(image, shape.extent(Axis[Width]), shape.extent(Axis[Height]))
                                              ^^^^^^^^^^^
```

## Why it works

Two barriers, and unlike case 04 the second one does not depend on the author keeping an
extent wrapped:

1. **Typed extents.** `letterbox` takes `AxisExtent[Height]` and `AxisExtent[Width]`, so
   supplying them in the wrong order is a type error. Same mechanism as case 04.
2. **The layout is a type.** `type ModelInput = Tensor4[Batch, Channel, Height, Width,
   Float32]` states NCHW once. Even if the extents had been unwrapped to `Int`, a
   letterboxed buffer built as `Tensor3[Width, Height, Channel]` would be rejected by
   `toModelInput`, because `transpose((Axis[Batch], Axis[Channel], Axis[Height],
   Axis[Width]))` cannot reorder a tensor whose axes are the other way round.

That second barrier is what makes this case stronger than case 04. The defect propagates
into a tensor, and once it is in a tensor, DimWit's types see it regardless of whether the
extents happen to be equal.

## Contrast with the other tools

* jaxtyping catches it too — at run time, and only for rectangular models. The default
  square configuration slips past.
* DimWit rejects the program before it runs, and does so identically for 640x640 and
  640x480, because `Height` and `Width` are distinct types whatever their extents.

## Honest limits

* The guarantee rests on the pipeline being written against the `ModelInput` alias rather
  than against an untyped array handed over the ONNX boundary. At the FFI edge — where the
  bytes actually leave for onnxruntime — DimWit's types stop and a convention takes over,
  exactly as everywhere else. Interop boundaries are unprotected in every system here.
