//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 07 (fixed) — stacking chain and draw, DimWit.
  *
  * Flattening two axes does not invent a name: it produces the structured label
  * `Chain |*| Draw`. A user axis called `Sample` is a different type, so the two cannot
  * collide however they are spelled.
  */
object Case07Fixed:

  import dimwit.*

  trait Chain derives Label
  trait Draw derives Label

  /** The user's own dimension, which in the upstream bug shared the name `sample`. */
  trait Sample derives Label

  /** Flattened posterior draws. The axis records where it came from, in its type. */
  type Flattened = Tensor2[Chain |*| Draw, Sample, Float32]

  def flattenPosterior(posterior: Tensor3[Chain, Draw, Sample, Float32]): Flattened =
    posterior.flatten((Axis[Chain], Axis[Draw]))

  /** Averages over the MCMC draws, not over the user's samples. */
  def posteriorMean(flat: Flattened): Tensor1[Sample, Float32] =
    flat.mean(Axis[Chain |*| Draw])

  /** Averages over the user's samples, not over the MCMC draws. */
  def sampleMean(flat: Flattened): Tensor1[Chain |*| Draw, Float32] =
    flat.mean(Axis[Sample])

  @main def case07Check(): Unit =
    dimwit.initialize()

    val posterior = Tensor(Shape(Axis[Chain] -> 4, Axis[Draw] -> 3, Axis[Sample] -> 5))
      .fromArray(Array.tabulate(4 * 3 * 5)(_.toFloat))

    val flat = flattenPosterior(posterior)
    assert(flat.shape(Axis[Chain |*| Draw]) == 12, "chain and draw must fuse into 12")
    assert(flat.shape(Axis[Sample]) == 5, "the user's Sample axis must survive untouched")
    assert(posteriorMean(flat).shape(Axis[Sample]) == 5)
    assert(sampleMean(flat).shape(Axis[Chain |*| Draw]) == 12)
    println("case07 ok: Chain |*| Draw and Sample stayed distinct")
