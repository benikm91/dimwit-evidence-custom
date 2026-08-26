//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 14 (buggy) — Atari preprocessing, DimWit.
  *
  * THIS FILE COMPILES. That is the finding, not a harness failure.
  *
  * The upstream defect is one integer standing for two different concepts. DimWit cannot
  * object: `Axis[Height] -> screenSize` and `Axis[Width] -> screenSize` are both well-typed
  * for any `screenSize`, and they have to be, since square tensors are legitimate.
  *
  * What DimWit does provide is that the assumption is legible — the reuse of `screenSize`
  * sits between two named axes rather than inside an anonymous `(size, size)` tuple.
  */
object Case14Buggy:

  import dimwit.*

  trait Height derives Label
  trait Width derives Label

  /** `screen_size: int` — one number, silently used for both spatial axes. */
  def preprocess(
      frame: Tensor2[Height, Width, Float32],
      screenSize: Int
  ): Tensor2[Height, Width, Float32] =
    val rows = (0 until screenSize).map(r => r * frame.shape(Axis[Height]) / screenSize)
    val cols = (0 until screenSize).map(c => c * frame.shape(Axis[Width]) / screenSize)
    frame
      .slice(Axis[Height].at(rows))
      .slice(Axis[Width].at(cols))

  @main def case14BuggyDemo(): Unit =
    dimwit.initialize()

    val frame = Tensor(Shape(Axis[Height] -> 210, Axis[Width] -> 160))
      .fromArray(Array.tabulate(210 * 160)(_.toFloat))

    val out = preprocess(frame, 84)
    assert(out.shape(Axis[Height]) == out.shape(Axis[Width]), "expected a forced square")
    println(s"case14 buggy: compiled and ran; 210x160 became ${out.shape(Axis[Height])}x${out.shape(Axis[Width])}")
