//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 13 (buggy) — mel spectrogram, DimWit. DOES NOT COMPILE, on purpose.
  *
  * The extractor is the one from `Fixed.scala`; only the call is wrong. Upstream a
  * `[batch, samples]` array was handed to a function written for one waveform, and the
  * framing step sliced across clips instead of across time.
  */
object Case13Buggy:

  import dimwit.*
  import Case13Fixed.{Batch, Frame, Mel, Sample, Window, logMelSingle}

  def logMelBatchedBuggy(
      batch: Tensor2[Batch, Sample, Float32],
      filters: Tensor2[Window, Mel, Float32]
  ): Tensor2[Frame, Mel, Float32] =
    // a batch of clips passed straight into the single-waveform extractor => compile-error
    logMelSingle(batch, filters)
