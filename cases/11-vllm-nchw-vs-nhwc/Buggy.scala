//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 11 (buggy) — video frame layout, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of the vLLM defect: the NCHW backend's output is handed straight to the
  * NHWC pipeline. In NumPy the two are the same object when height, width and channels all
  * equal 3; in DimWit they are different types whatever the extents.
  *
  * Expected compiler error:
  *
  *   Found:    Case11Buggy.Nchw  (Tensor[(Frame, Channel, Height, Width), Float32])
  *   Required: Case11Buggy.Nhwc  (Tensor[(Frame, Height, Width, Channel), Float32])
  */
object Case11Buggy:

  import dimwit.*

  trait Frame derives Label
  trait Channel derives Label
  trait Height derives Label
  trait Width derives Label

  type Nchw = Tensor4[Frame, Channel, Height, Width, Float32]
  type Nhwc = Tensor4[Frame, Height, Width, Channel, Float32]

  def perChannelMean(frames: Nhwc): Tensor1[Channel, Float32] =
    frames.mean((Axis[Frame], Axis[Height], Axis[Width]))

  def normalise(decoded: Nchw): Tensor1[Channel, Float32] =
    // The backend was swapped and nobody transposed.
    perChannelMean(decoded)
