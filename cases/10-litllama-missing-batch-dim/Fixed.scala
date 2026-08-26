//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 10 (fixed) — next-token logits, DimWit.
  *
  * The model is written for ONE sequence. `Batch` never appears in its signature, so it
  * cannot be forgotten, mis-placed or double-counted. Batching is `vmap`, from the outside.
  */
object Case10Fixed:

  import dimwit.*

  trait Batch derives Label
  trait Pos derives Label
  trait Embed derives Label
  trait Vocab derives Label

  /** One sequence in, one row of logits per position out. */
  def model(
      embeddings: Tensor2[Pos, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor2[Pos, Vocab, Float32] =
    // centring across positions, stated as such rather than as `axis=0`
    val centred = embeddings -! embeddings.mean(Axis[Pos])
    centred.vmap(Axis[Pos])(row => row.dot(Axis[Embed])(wOut))

  /** The logits for the position we are about to extend. Named, not `-1`. */
  def nextTokenLogits(
      embeddings: Tensor2[Pos, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor1[Vocab, Float32] =
    val logits = model(embeddings, wOut)
    logits.slice(Axis[Pos].at(logits.shape(Axis[Pos]) - 1))

  /** Batched generation, if it is wanted, is a lift of the single-sequence function. */
  def batchedLogits(
      embeddings: Tensor3[Batch, Pos, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor2[Batch, Vocab, Float32] =
    embeddings.vmap(Axis[Batch])(seq => nextTokenLogits(seq, wOut))

  @main def case10Check(): Unit =
    dimwit.initialize()

    val wOut = Tensor(Shape(Axis[Embed] -> 4, Axis[Vocab] -> 7)).fill(0.5f)
    val one = Tensor(Shape(Axis[Pos] -> 4, Axis[Embed] -> 4)).fromArray(Array.tabulate(16)(_.toFloat))

    assert(nextTokenLogits(one, wOut).shape(Axis[Vocab]) == 7)

    val many = Tensor(Shape(Axis[Batch] -> 2, Axis[Pos] -> 4, Axis[Embed] -> 4))
      .fromArray(Array.tabulate(32)(_.toFloat))
    val batched = batchedLogits(many, wOut)
    assert(batched.shape(Axis[Batch]) == 2 && batched.shape(Axis[Vocab]) == 7)
    println("case10 ok: single-sequence model, batching added by vmap")
