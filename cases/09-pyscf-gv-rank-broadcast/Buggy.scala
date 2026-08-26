//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 09 (buggy) — DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of `ft_ao(mol, g)` where `g` is a single `(3,)` vector and the routine
  * expects `(N, 3)`. NumPy prepends the missing axis silently; DimWit does not have a
  * missing axis to prepend, because rank is part of the type.
  *
  * Expected compiler error:
  *
  *   Found:    Tensor1[Case09Buggy.Component, Float32]
  *   Required: Tensor[(Case09Buggy.GPoint, Case09Buggy.Component), Float32]
  */
object Case09Buggy:

  import dimwit.*

  trait GPoint derives Label
  trait Component derives Label
  trait Center derives Label

  def ftAoSingle(
      g: Tensor1[Component, Float32],
      centers: Tensor1[Center, Float32]
  ): Tensor1[Center, Float32] =
    val phase = (g * g).sum * Tensor0(-0.5f)
    centers *! phase.exp

  def ftAo(
      gv: Tensor2[GPoint, Component, Float32],
      centers: Tensor1[Center, Float32]
  ): Tensor2[GPoint, Center, Float32] =
    gv.vmap(Axis[GPoint])(g => ftAoSingle(g, centers))

  def transformOnePoint(
      g: Tensor1[Component, Float32],
      centers: Tensor1[Center, Float32]
  ): Tensor2[GPoint, Center, Float32] =
    // The caller has one vector and hands it to the grid routine, exactly as in the issue.
    ftAo(g, centers)
