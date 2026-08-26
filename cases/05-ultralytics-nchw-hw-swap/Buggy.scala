//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 05 (buggy) — YOLO ONNX preprocessing, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of
  *   self.input_width  = input_shape[2]
  *   self.input_height = input_shape[3]
  * The author reads the NCHW spatial extents in the wrong order and letterboxes into a
  * transposed buffer.
  *
  * Expected compiler errors: `AxisExtent[Width]` supplied where `AxisExtent[Height]` is
  * required (and vice versa), and the transposed buffer rejected by `toModelInput`.
  */
object Case05Buggy:

  import dimwit.*

  trait Batch derives Label
  trait Channel derives Label
  trait Height derives Label
  trait Width derives Label

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

  def preprocess(
      image: Tensor3[Height, Width, Channel, Float32],
      modelInput: ModelInput
  ): ModelInput =
    val shape = modelInput.shape
    // input_shape[2] was read as the width and input_shape[3] as the height.
    val boxed = letterbox(image, shape.extent(Axis[Width]), shape.extent(Axis[Height]))
    toModelInput(boxed)
