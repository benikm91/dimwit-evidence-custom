//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 13 (buggy) — mel spectrogram, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of `log_mel_spectrogram(batch)`: a `[batch, samples]` array handed to a
  * function written for one waveform.
  *
  * Expected compiler error:
  *
  *   Found:    Tensor2[Case13Buggy.Batch, Case13Buggy.Sample, Float32]
  *   Required: Tensor1[Case13Buggy.Sample, Float32]
  */
object Case13Buggy:

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

  def features(
      batch: Tensor2[Batch, Sample, Float32],
      filters: Tensor2[Window, Mel, Float32]
  ): Tensor2[Frame, Mel, Float32] =
    logMel(batch, filters)
