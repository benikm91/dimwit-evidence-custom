//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 11 (fixed) — video frame layout, DimWit.
  *
  * The two layouts are two types. A backend that produces NCHW cannot be plugged into a
  * pipeline that consumes NHWC without an explicit transpose, and the transpose is written
  * with axis names so it cannot itself be got wrong.
  */
object Case11Fixed:

  import dimwit.*

  trait Frame derives Label
  trait Channel derives Label
  trait Height derives Label
  trait Width derives Label

  type Nchw = Tensor4[Frame, Channel, Height, Width, Float32]
  type Nhwc = Tensor4[Frame, Height, Width, Channel, Float32]

  def toNhwc(frames: Nchw): Nhwc =
    frames.transpose((Axis[Frame], Axis[Height], Axis[Width], Axis[Channel]))

  /** One mean per channel. Which axes are reduced is stated, not implied by position. */
  def perChannelMean(frames: Nhwc): Tensor1[Channel, Float32] =
    frames.mean((Axis[Frame], Axis[Height], Axis[Width]))

  @main def case11Check(): Unit =
    dimwit.initialize()

    // the silent case upstream: square frames with three channels
    val decoded: Nchw = Tensor(
      Shape(Axis[Frame] -> 2, Axis[Channel] -> 3, Axis[Height] -> 3, Axis[Width] -> 3)
    ).fromArray(Array.tabulate(2 * 3 * 3 * 3)(_.toFloat))

    val stats = perChannelMean(toNhwc(decoded))
    assert(stats.shape(Axis[Channel]) == 3, "one mean per channel")
    // channel c contains frame-major values, so its mean is computable by hand
    val expected = Array(4.0f + 13.5f, 7.0f + 13.5f, 10.0f + 13.5f)
    val got = stats.toArray
    assert(got.zip(expected).forall((a, b) => math.abs(a - b) < 1e-3f), s"got ${got.mkString(",")}")
    println("case11 ok: per-channel statistics reduced over Frame, Height, Width")
