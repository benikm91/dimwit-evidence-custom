//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 08 (buggy) — flattening the sampling dimensions, DimWit. DOES NOT COMPILE.
  *
  * The upstream defect fused the axes using information read off the dataset BEFORE it was
  * transposed. Transliterated: the author flattens in the dataset's own order
  * `(Draw, Chain)` while the rest of the program is written against `Chain |*| Draw`.
  *
  * Expected compiler error:
  *
  *   Found:    Tensor[(Case08Buggy.Draw |*| Case08Buggy.Chain, Case08Buggy.Team), Float32]
  *   Required: Tensor[(Case08Buggy.Chain |*| Case08Buggy.Draw, Case08Buggy.Team), Float32]
  */
object Case08Buggy:

  import dimwit.*

  trait Chain derives Label
  trait Draw derives Label
  trait Team derives Label

  type PointList = Tensor2[Chain |*| Draw, Team, Float32]

  def toPointList(ds: Tensor3[Team, Draw, Chain, Float32]): PointList =
    // The axes are moved to the front in the order they appear in the dataset
    // ("team", "draw", "chain") rather than in the order the point list requires.
    val ordered = ds.transpose((Axis[Draw], Axis[Chain], Axis[Team]))
    ordered.flatten((Axis[Draw], Axis[Chain]))
