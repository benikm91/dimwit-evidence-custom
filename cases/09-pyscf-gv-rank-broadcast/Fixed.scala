//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 09 (fixed) — Fourier transform over a grid of G-vectors, DimWit.
  *
  * A single G-vector is `Tensor1[Component, Float32]`; a grid of them is
  * `Tensor2[GPoint, Component, Float32]`. The two are different types, and the grid
  * version is obtained by mapping the single-vector version — never by hoping that
  * broadcasting does the right thing.
  */
object Case09Fixed:

  import dimwit.*

  /** Reciprocal-space grid point. */
  trait GPoint derives Label

  /** x / y / z of a reciprocal-space vector. */
  trait Component derives Label

  /** Basis-function centre. */
  trait Center derives Label

  /** The physics, for ONE G-vector. */
  def ftAoSingle(
      g: Tensor1[Component, Float32],
      centers: Tensor1[Center, Float32]
  ): Tensor1[Center, Float32] =
    val phase = (g * g).sum * Tensor0(-0.5f)
    centers *! phase.exp

  /** The grid version is the single version, mapped. */
  def ftAo(
      gv: Tensor2[GPoint, Component, Float32],
      centers: Tensor1[Center, Float32]
  ): Tensor2[GPoint, Center, Float32] =
    gv.vmap(Axis[GPoint])(g => ftAoSingle(g, centers))

  @main def case09Check(): Unit =
    dimwit.initialize()

    val centers = Tensor1(Axis[Center]).fromArray(Array(0f, 0.5f, 1f, 1.5f))
    val grid = Tensor(Shape(Axis[GPoint] -> 3, Axis[Component] -> 3))
      .fromArray(Array(0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 1f))

    val onGrid = ftAo(grid, centers)
    assert(onGrid.shape(Axis[GPoint]) == 3 && onGrid.shape(Axis[Center]) == 4)

    // A single vector goes through the single-vector entry point. No rank is guessed.
    val one = grid.slice(Axis[GPoint].at(1))
    assert(ftAoSingle(one, centers).shape(Axis[Center]) == 4)
    println("case09 ok: one G-vector and a grid of them have different types")
