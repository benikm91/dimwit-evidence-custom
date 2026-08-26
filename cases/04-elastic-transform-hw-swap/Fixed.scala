//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 04 (fixed) — elastic displacement normalisation, DimWit.
  *
  * The point of this case is `AxisExtent[L]`. `shape.extent(Axis[Width])` returns an
  * `AxisExtent[Width]`, which is a *different type* from `AxisExtent[Height]`. As long as
  * the divisor stays wrapped, handing the height to a function that wants the width is a
  * type error.
  */
object Case04Fixed:

  import dimwit.*

  trait Height derives Label
  trait Width derives Label

  /** Horizontal displacement is normalised into [-1, 1] by the WIDTH. */
  def normaliseHorizontal(
      dx: Tensor2[Height, Width, Float32],
      alpha: Float,
      width: AxisExtent[Width]
  ): Tensor2[Height, Width, Float32] =
    dx *! Tensor0(alpha / width.size.toFloat)

  /** Vertical displacement is normalised into [-1, 1] by the HEIGHT. */
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
      normaliseHorizontal(dx, alpha, shape.extent(Axis[Width])),
      normaliseVertical(dy, alpha, shape.extent(Axis[Height]))
    )

  @main def case04Check(): Unit =
    dimwit.initialize()

    // the reporter's 100 x 800 image
    val ones = Tensor(Shape(Axis[Height] -> 100, Axis[Width] -> 800)).fill(1.0f)
    val (hx, vy) = displacementField(ones, ones, alpha = 50.0f)

    val horizontal = hx.slice(Axis[Height].at(0)).slice(Axis[Width].at(0)).item
    val vertical = vy.slice(Axis[Height].at(0)).slice(Axis[Width].at(0)).item

    assert(math.abs(horizontal - 50.0f / 800f) < 1e-7f, s"horizontal must use the width, got $horizontal")
    assert(math.abs(vertical - 50.0f / 100f) < 1e-7f, s"vertical must use the height, got $vertical")
    println("case04 ok: horizontal normalised by Width, vertical by Height")
