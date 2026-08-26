//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 12 (buggy) — batched contraction, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of `tensordot(a, b, axes=([1], [0]))`: the author contracts the row axis
  * instead of the inner one. In NumPy that is legal whenever the two happen to have the
  * same extent. In DimWit `Row` does not occur in `b`, so there is no contraction to form.
  *
  * Expected compiler error: no `AxisRemover` for `Row` in `(Inner, Col)`.
  */
object Case12Buggy:

  import dimwit.*

  trait Batch derives Label
  trait Row derives Label
  trait Inner derives Label
  trait Col derives Label

  def matmul(
      a: Tensor2[Row, Inner, Float32],
      b: Tensor2[Inner, Col, Float32]
  ): Tensor2[Row, Col, Float32] =
    // Contracting Row: `b` has no Row axis to contract against.
    a.dot(Axis[Row])(b)
