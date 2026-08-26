# DimWit — verdict

**Category: `compile-time detection`, conditional on keeping the extent wrapped**

Verified: `scala-cli compile Buggy.scala` fails with

```
Found:    dimwit.tensor.Axis[Case04Buggy.Height]
Required: dimwit.tensor.Axis[Case04Buggy.Width]
      normaliseHorizontal(dx, alpha, shape.extent(Axis[Height])),
                                                  ^^^^^^^^^^^^
```

## Why it works

Two things have to be true, and only the first is automatic:

1. **`shape.extent(Axis[L])` returns `AxisExtent[L]`, not `Int`.** The extent keeps the
   identity of the axis it came from, so `AxisExtent[Height]` and `AxisExtent[Width]` are
   incompatible types. This is the mechanism.
2. **The API has to accept the wrapped extent.** `normaliseHorizontal(..., width:
   AxisExtent[Width])` is what makes the mistake statable and therefore checkable.

## The escape hatch, stated plainly

DimWit also offers `shape(Axis[Height]): Int`. Written that way:

```scala
val h = dx.shape(Axis[Height])   // Int
val w = dx.shape(Axis[Width])    // Int
dx *! Tensor0(alpha / h.toFloat) // the bug, and it compiles
```

the extent has been unwrapped and DimWit is exactly as blind as NumPy. So this row is
**not** a claim that DimWit catches height/width confusion in general. The honest claim is
narrower and still worth making:

> DimWit removes *positional* shape access. `size[0]` cannot be written. The author must
> name the axis, and if they keep the named extent wrapped the compiler checks the choice.

The upstream bug was `size[0]` vs `size[1]` — an index slip, invisible in review. Its DimWit
counterpart is `Axis[Height]` vs `Axis[Width]` — a naming mistake, visible in review even
before the compiler weighs in. That is a real improvement in the failure mode even in the
unwrapped case, but it is a *legibility* argument, not a soundness one, and the paper should
label it as such.

## Suggested framing for the table

Score this row `compile-time detection` with a footnote, or score it `partly prevented` if
the table has such a column. Do not score it as an unqualified win.
