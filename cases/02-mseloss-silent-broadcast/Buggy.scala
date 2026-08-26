//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 02 (buggy) — mean squared error, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of `nn.Linear(2, 1)` followed by `nn.MSELoss()(pred, target)`: the
  * model keeps an explicit width-1 output axis, so predictions are `[Batch, Out]` while
  * the targets are `[Batch]`.
  *
  * Expected compiler error:
  *
  *   Found:    Tensor1[Case02Buggy.Batch, Float32]
  *   Required: Tensor[(Case02Buggy.Batch, Case02Buggy.Out), Float32]
  */
object Case02Buggy:

  import dimwit.*

  trait Batch derives Label
  trait Feature derives Label

  /** The `out_features = 1` axis of `nn.Linear(2, 1)`, kept explicitly. */
  trait Out derives Label

  def predict(x: Tensor1[Feature, Float32], w: Tensor2[Feature, Out, Float32]): Tensor1[Out, Float32] =
    x.dot(Axis[Feature])(w)

  def loss(
      xs: Tensor2[Batch, Feature, Float32],
      ys: Tensor1[Batch, Float32],
      w: Tensor2[Feature, Out, Float32]
  ): Tensor0[Float32] =
    val preds: Tensor2[Batch, Out, Float32] = xs.vmap(Axis[Batch])(x => predict(x, w))
    // In PyTorch this line broadcasts [n, 1] against [n] into [n, n] and says nothing.
    // In DimWit `-` is shape-exact, so it does not type check. Asking for the broadcast
    // explicitly (`preds -! ys`) would compile — and would be a visible decision.
    val residuals = preds - ys
    (residuals * residuals).mean
