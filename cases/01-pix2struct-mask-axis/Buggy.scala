//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 01 (buggy) — Pix2Struct attention mask, DimWit. DOES NOT COMPILE, on purpose.
  *
  * The Python defect was `attention_mask[:, None, :, None]`: a mask over KEY positions
  * placed on the QUERY axis. Transliterated, the author has written a helper that masks
  * target (query) positions and feeds it the source padding mask.
  *
  * Expected compiler error at the call site in `attend`:
  *
  *   Found:    Tensor1[Case01Buggy.Source, Bool]
  *   Required: Tensor1[Case01Buggy.Target, Bool]
  */
object Case01Buggy:

  import dimwit.*
  import dimwit.nn.ActivationFunctions.softmax

  trait Target derives Label
  trait Source derives Label

  def sourcePaddingMask(sourceTokens: Tensor1[Source, Int32]): Tensor1[Source, Bool] =
    sourceTokens > Tensor.like(sourceTokens).fill(0)

  /** The transliteration of `mask[:, None, :, None]`: the mask lands on the query axis. */
  def maskTargetPositions(
      scores: Tensor2[Target, Source, Float32],
      mask: Tensor1[Target, Bool]
  ): Tensor2[Target, Source, Float32] =
    val masked = where_!(mask, scores, Tensor0(Float.NegativeInfinity))
    masked.vapply(Axis[Source])(softmax)

  def attend(
      scores: Tensor2[Target, Source, Float32],
      sourceTokens: Tensor1[Source, Int32]
  ): Tensor2[Target, Source, Float32] =
    // A mask over SOURCE positions cannot be used to mask TARGET positions.
    maskTargetPositions(scores, sourcePaddingMask(sourceTokens))
