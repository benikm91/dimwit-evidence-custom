//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 04 (buggy) — elastic displacement normalisation, DimWit. DOES NOT COMPILE.
  *
  * Transliteration of `dx * alpha[0] / size[0]`: the horizontal displacement is normalised
  * by the height. In DimWit there is no positional `size[0]`; the author has to name the
  * axis, and naming the wrong one is rejected because `AxisExtent[Height]` is not
  * `AxisExtent[Width]`.
  *
  * Expected compiler error:
  *
  *   Found:    AxisExtent[Case04Buggy.Height]
  *   Required: AxisExtent[Case04Buggy.Width]
  */
object Case04Buggy:

  import dimwit.*

  trait Height derives Label
  trait Width derives Label

  def normaliseHorizontal(
      dx: Tensor2[Height, Width, Float32],
      alpha: Float,
      width: AxisExtent[Width]
  ): Tensor2[Height, Width, Float32] =
    dx *! Tensor0(alpha / width.size.toFloat)

  def normaliseVertical(
      dy: Tensor2[Height, Width, Float32],
      alpha: Float,
      height: AxisExtent[Height]
  ): Tensor2[Height, Width, Float32] =
    dy *! Tensor0(alpha / height.size.toFloat)

  def displacementField(
      dx: Tensor2[Height, Width, Float32],
      dy: Tensor2[Height, Width, Float32],
      alpha: Float
  ): (Tensor2[Height, Width, Float32], Tensor2[Height, Width, Float32]) =
    val shape = dx.shape
    (
      // `size[0]` is the height. Handing it to the horizontal normaliser is the bug.
      normaliseHorizontal(dx, alpha, shape.extent(Axis[Height])),
      normaliseVertical(dy, alpha, shape.extent(Axis[Width]))
    )
