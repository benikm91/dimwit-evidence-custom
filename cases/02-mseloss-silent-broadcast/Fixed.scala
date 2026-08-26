//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 02 (fixed) — mean squared error, DimWit.
  *
  * The model is written for ONE example and returns a scalar. Batching it with `vmap`
  * produces `Tensor1[Batch, Float32]`, exactly the shape of the targets, so `-` applies
  * without any broadcast at all.
  */
object Case02Fixed:

  import dimwit.*

  trait Batch derives Label
  trait Feature derives Label

  /** One example in, one number out. There is no room for a width-1 output axis. */
  def predict(x: Tensor1[Feature, Float32], w: Tensor1[Feature, Float32], b: Tensor0[Float32]): Tensor0[Float32] =
    x.dot(Axis[Feature])(w) + b

  def loss(
      xs: Tensor2[Batch, Feature, Float32],
      ys: Tensor1[Batch, Float32],
      w: Tensor1[Feature, Float32],
      b: Tensor0[Float32]
  ): Tensor0[Float32] =
    val preds: Tensor1[Batch, Float32] = xs.vmap(Axis[Batch])(x => predict(x, w, b))
    val residuals = preds - ys // shape-exact: no broadcast is possible here
    (residuals * residuals).mean

  @main def case02Check(): Unit =
    dimwit.initialize()

    // y = 2*x0 + 3*x1, four examples, exact fit
    val xs = Tensor(Shape(Axis[Batch] -> 4, Axis[Feature] -> 2))
      .fromArray(Array(1f, 0f, 0f, 1f, 1f, 1f, 2f, 1f))
    val ys = Tensor1(Axis[Batch]).fromArray(Array(2f, 3f, 5f, 7f))
    val w = Tensor1(Axis[Feature]).fromArray(Array(2f, 3f))
    val b = Tensor0(0f)

    val l = loss(xs, ys, w, b).item
    assert(l < 1e-6f, s"exact parameters must give zero loss, got $l")
    println(s"case02 ok: loss with the true parameters is $l")
