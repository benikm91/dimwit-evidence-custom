//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 14 (fixed) — Atari preprocessing, DimWit.
  *
  * The target size is two typed extents, so a caller who wants a rectangle can ask for one
  * and a caller who wants a square has to say `Axis[Height] -> n, Axis[Width] -> n`
  * explicitly. Note that this is an API improvement, not a type-system guarantee: see
  * Buggy.scala, which compiles.
  */
object Case14Fixed:

  import dimwit.*

  trait Height derives Label
  trait Width derives Label

  def preprocess(
      frame: Tensor2[Height, Width, Float32],
      height: AxisExtent[Height],
      width: AxisExtent[Width]
  ): Tensor2[Height, Width, Float32] =
    val rows = (0 until height.size).map(r => r * frame.shape(Axis[Height]) / height.size)
    val cols = (0 until width.size).map(c => c * frame.shape(Axis[Width]) / width.size)
    frame
      .slice(Axis[Height].at(rows))
      .slice(Axis[Width].at(cols))

  @main def case14Check(): Unit =
    dimwit.initialize()

    val frame = Tensor(Shape(Axis[Height] -> 210, Axis[Width] -> 160))
      .fromArray(Array.tabulate(210 * 160)(_.toFloat))

    val rect = preprocess(frame, Axis[Height] -> 84, Axis[Width] -> 64)
    assert(rect.shape(Axis[Height]) == 84 && rect.shape(Axis[Width]) == 64, "aspect ratio preserved")

    val square = preprocess(frame, Axis[Height] -> 84, Axis[Width] -> 84)
    assert(square.shape(Axis[Height]) == 84 && square.shape(Axis[Width]) == 84)
    println("case14 ok: rectangular targets expressible; squares must be asked for by name")
