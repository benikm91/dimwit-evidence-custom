//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 03 (buggy) — cross product, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of `torch.cross(dim=None)`: the author batches over the wrong axis,
  * i.e. maps over `Component` and treats the `Sample` slices as 3-vectors. That is exactly
  * what the torch backend did when it picked the first length-3 axis.
  *
  * Expected compiler error:
  *
  *   Found:    Tensor1[Case03Buggy.Sample, Float32]
  *   Required: Tensor1[Case03Buggy.Component, Float32]
  */
object Case03Buggy:

  import dimwit.*

  trait Sample derives Label
  trait Component derives Label

  private def at(t: Tensor1[Component, Float32], i: Int): Tensor0[Float32] =
    t.slice(Axis[Component].at(i))

  def cross3(a: Tensor1[Component, Float32], b: Tensor1[Component, Float32]): Tensor1[Component, Float32] =
    val (a0, a1, a2) = (at(a, 0), at(a, 1), at(a, 2))
    val (b0, b1, b2) = (at(b, 0), at(b, 1), at(b, 2))
    stack(
      Seq(a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0),
      Axis[Component]
    )

  def crossBatched(
      a: Tensor2[Sample, Component, Float32],
      b: Tensor2[Sample, Component, Float32]
  ): Tensor2[Sample, Component, Float32] =
    // Mapping over Component leaves Sample slices, which are not 3-vectors.
    zipvmap(Axis[Component])(a, b)((x: Tensor1[Sample, Float32], y: Tensor1[Sample, Float32]) => cross3(x, y))
