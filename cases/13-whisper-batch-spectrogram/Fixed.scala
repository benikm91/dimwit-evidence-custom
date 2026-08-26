//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 13 (fixed) — mel spectrogram, DimWit.
  *
  * The extractor is typed for one waveform: `Tensor1[Sample, Float32]` in,
  * `Tensor2[Frame, Mel, Float32]` out. Batching is `vmap`, and the framing slices
  * `Axis[Sample]` by name, so it cannot slice a batch even if one is added later.
  */
object Case13Fixed:

  import dimwit.*

  trait Batch derives Label
  trait Sample derives Label
  trait Frame derives Label
  trait Window derives Label
  trait Mel derives Label

  private val WindowSize = 4
  private val Hop = 2

  def logMel(
      audio: Tensor1[Sample, Float32],
      filters: Tensor2[Window, Mel, Float32]
  ): Tensor2[Frame, Mel, Float32] =
    val n = (audio.shape(Axis[Sample]) - WindowSize) / Hop + 1
    val frames = (0 until n).map { i =>
      audio
        .slice(Axis[Sample].at(i * Hop until i * Hop + WindowSize))
        .relabelTo(Axis[Window])
    }
    val framed: Tensor2[Frame, Window, Float32] = stack(frames, Axis[Frame])
    framed.vmap(Axis[Frame])(w => w.dot(Axis[Window])(filters).abs.log)

  def logMelBatched(
      batch: Tensor2[Batch, Sample, Float32],
      filters: Tensor2[Window, Mel, Float32]
  ): Tensor3[Batch, Frame, Mel, Float32] =
    batch.vmap(Axis[Batch])(clip => logMel(clip, filters))

  @main def case13Check(): Unit =
    dimwit.initialize()

    val filters = Tensor(Shape(Axis[Window] -> 4, Axis[Mel] -> 3)).fill(1.0f)
    val audio = Tensor1(Axis[Sample]).fromArray(Array.tabulate(12)(i => (i + 1).toFloat))

    val single = logMel(audio, filters)
    assert(single.shape(Axis[Frame]) == 5 && single.shape(Axis[Mel]) == 3, "5 frames of 3 mels")

    val batch = Tensor(Shape(Axis[Batch] -> 2, Axis[Sample] -> 12))
      .fromArray(Array.tabulate(24)(i => (i + 1).toFloat))
    val many = logMelBatched(batch, filters)
    assert(many.shape(Axis[Batch]) == 2 && many.shape(Axis[Frame]) == 5)
    println("case13 ok: framing slices Sample, batching added by vmap")
