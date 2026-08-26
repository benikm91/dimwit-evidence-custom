//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 10 (buggy) — next-token logits, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of `logits = model(idx_cond)`: a batch of sequences handed to a model
  * written for one sequence, in the hope that everything is rank-polymorphic. In NumPy and
  * PyTorch it is. In DimWit the rank and the axis names are both in the type.
  *
  * Expected compiler error:
  *
  *   Found:    Tensor3[Case10Buggy.Batch, Case10Buggy.Pos, Case10Buggy.Embed, Float32]
  *   Required: Tensor2[Case10Buggy.Pos, Case10Buggy.Embed, Float32]
  */
object Case10Buggy:

  import dimwit.*

  trait Batch derives Label
  trait Pos derives Label
  trait Embed derives Label
  trait Vocab derives Label

  def model(
      embeddings: Tensor2[Pos, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor2[Pos, Vocab, Float32] =
    val centred = embeddings -! embeddings.mean(Axis[Pos])
    centred.vmap(Axis[Pos])(row => row.dot(Axis[Embed])(wOut))

  def generate(
      embeddings: Tensor3[Batch, Pos, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor2[Pos, Vocab, Float32] =
    // The batched tensor is passed straight into the single-sequence model.
    model(embeddings, wOut)
