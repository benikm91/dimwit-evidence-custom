//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 05 (fixed) — YOLO ONNX preprocessing, DimWit.
  *
  * The letterboxed image is `Tensor3[Height, Width, Channel]` and the model input is
  * `Tensor4[Batch, Channel, Height, Width]`. The NCHW layout is written once, in a type,
  * and every producer has to agree with it.
  */
object Case05Fixed:

  import dimwit.*

  trait Batch derives Label
  trait Channel derives Label
  trait Height derives Label
  trait Width derives Label

  /** The model's declared input. NCHW is in the type, not in a comment. */
  type ModelInput = Tensor4[Batch, Channel, Height, Width, Float32]

  def letterbox(
      image: Tensor3[Height, Width, Channel, Float32],
      height: AxisExtent[Height],
      width: AxisExtent[Width]
  ): Tensor3[Height, Width, Channel, Float32] =
    val channel = image.shape.extent(Axis[Channel])
    Tensor(Shape(height, width, channel)).fill(0.0f)

  def toModelInput(letterboxed: Tensor3[Height, Width, Channel, Float32]): ModelInput =
    letterboxed
      .appendAxis(Axis[Batch])
      .transpose((Axis[Batch], Axis[Channel], Axis[Height], Axis[Width]))

  @main def case05Check(): Unit =
    dimwit.initialize()

    val image = Tensor(Shape(Axis[Height] -> 1080, Axis[Width] -> 1920, Axis[Channel] -> 3)).fill(0.0f)
    val boxed = letterbox(image, Axis[Height] -> 640, Axis[Width] -> 480)
    val input: ModelInput = toModelInput(boxed)

    assert(input.shape(Axis[Height]) == 640, s"height must be 640, got ${input.shape(Axis[Height])}")
    assert(input.shape(Axis[Width]) == 480, s"width must be 480, got ${input.shape(Axis[Width])}")
    println("case05 ok: NCHW input built as (Batch, Channel, Height, Width) = (1, 3, 640, 480)")
