//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 06 (buggy) — softmax over the batch axis, DimWit.
  *
  * THIS FILE COMPILES. That is the finding, not a mistake in the harness.
  *
  * `vapply(Axis[Batch])(softmax)` and `vapply(Axis[Class])(softmax)` are both total,
  * well-typed functions `Tensor2[Batch, Class, Float32] => Tensor2[Batch, Class, Float32]`.
  * Choosing the wrong reduction axis does not change any shape, so no shape-based type
  * discipline — DimWit's included — can reject it.
  *
  * What DimWit does provide is that the mistake is spelled `Axis[Batch]` rather than
  * `dim=0`. See DIMWIT_SOLUTION.md.
  */
object Case06Buggy:

  import dimwit.*
  import dimwit.nn.ActivationFunctions.softmax

  trait Batch derives Label
  trait Class derives Label

  def probabilities(logits: Tensor2[Batch, Class, Float32]): Tensor2[Batch, Class, Float32] =
    // Normalises across examples instead of across classes.
    logits.vapply(Axis[Batch])(softmax)

  @main def case06BuggyDemo(): Unit =
    dimwit.initialize()

    val logits = Tensor(Shape(Axis[Batch] -> 4, Axis[Class] -> 4))
      .fromArray(Array.tabulate(16)(_.toFloat))

    val p = probabilities(logits)
    val rowSums = p.sum(Axis[Class]).toArray
    assert(!rowSums.forall(s => math.abs(s - 1.0f) < 1e-5f), "expected the rows NOT to be distributions")
    println(s"case06 buggy: compiled and ran; row sums are ${rowSums.mkString(", ")} instead of 1")
