//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 01 (fixed) — Pix2Struct attention mask, DimWit.
  *
  * The padding mask carries one flag per SOURCE (key) position, so its type is
  * `Tensor1[Source, Bool]`. Broadcasting it against `Tensor2[Target, Source, Float32]`
  * can only align it with `Source`. There is no positional axis insertion to get wrong,
  * which is why the bug of transformers#23974 has no counterpart here.
  */
object Case01Fixed:

  import dimwit.*
  import dimwit.nn.ActivationFunctions.softmax

  /** Query positions of the attending sequence. */
  trait Target derives Label

  /** Key positions of the attended sequence. */
  trait Source derives Label

  /** true = real token, false = padding. One flag per SOURCE position. */
  def sourcePaddingMask(sourceTokens: Tensor1[Source, Int32]): Tensor1[Source, Bool] =
    sourceTokens > Tensor.like(sourceTokens).fill(0)

  /** Mask out padded KEY positions, then normalise over the keys. */
  def maskedAttentionWeights(
      scores: Tensor2[Target, Source, Float32],
      keyMask: Tensor1[Source, Bool]
  ): Tensor2[Target, Source, Float32] =
    val negInf = Tensor0(Float.NegativeInfinity)
    val masked = where_!(keyMask, scores, negInf)
    masked.vapply(Axis[Source])(softmax)

  @main def case01Check(): Unit =
    dimwit.initialize()

    // three source positions, the last of which is padding (token id 0)
    val tokens = Tensor1(Axis[Source]).fromArray(Array(7, 9, 0))
    val scores = Tensor(Shape(Axis[Target] -> 3, Axis[Source] -> 3)).fill(0.0f)

    val weights = maskedAttentionWeights(scores, sourcePaddingMask(tokens))
    val onPadding = weights.slice(Axis[Source].at(2)).sum.item

    assert(math.abs(onPadding) < 1e-6f, s"padding must receive zero weight, got $onPadding")
    assert(math.abs(weights.sum.item - 3.0f) < 1e-4f, "each of the 3 query rows must sum to 1")
    println("case01 ok: padded key position receives zero attention weight")
