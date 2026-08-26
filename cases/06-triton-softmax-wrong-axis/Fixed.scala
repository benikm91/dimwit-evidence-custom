//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 06 (fixed) — softmax over the class axis, DimWit.
  *
  * DimWit has no `softmax(x, dim = n)`. `softmax` is `Tensor1[L, V] => Tensor1[L, V]` and
  * is lifted over a chosen axis with `vapply`, so the axis appears as a NAME at the call
  * site. That makes the choice legible. It does not make it checkable — see
  * DIMWIT_SOLUTION.md and Buggy.scala, which compiles.
  */
object Case06Fixed:

  import dimwit.*
  import dimwit.nn.ActivationFunctions.softmax

  trait Batch derives Label
  trait Class derives Label

  def probabilities(logits: Tensor2[Batch, Class, Float32]): Tensor2[Batch, Class, Float32] =
    logits.vapply(Axis[Class])(softmax)

  @main def case06Check(): Unit =
    dimwit.initialize()

    val logits = Tensor(Shape(Axis[Batch] -> 4, Axis[Class] -> 4))
      .fromArray(Array.tabulate(16)(_.toFloat))

    val p = probabilities(logits)
    // every row must be a distribution over Class
    val rowSums = p.sum(Axis[Class]).toArray
    assert(rowSums.forall(s => math.abs(s - 1.0f) < 1e-5f), s"rows must sum to 1, got ${rowSums.mkString(",")}")
    println("case06 ok: normalised over Class, rows sum to 1")
