//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 07 (buggy) — stacking chain and draw, DimWit. DOES NOT COMPILE, on purpose.
  *
  * The upstream defect: after stacking, the flattened axis and the user's own axis both
  * answered to the name `sample`, so a function meant for the user's samples silently
  * received the stacked one.
  *
  * Expected compiler error:
  *
  *   Found:    Tensor1[Case07Buggy.Chain |*| Case07Buggy.Draw, Float32]
  *   Required: Tensor1[Case07Buggy.Sample, Float32]
  */
object Case07Buggy:

  import dimwit.*

  trait Chain derives Label
  trait Draw derives Label
  trait Sample derives Label

  def flattenPosterior(
      posterior: Tensor3[Chain, Draw, Sample, Float32]
  ): Tensor2[Chain |*| Draw, Sample, Float32] =
    posterior.flatten((Axis[Chain], Axis[Draw]))

  /** A downstream routine written against the USER's sample dimension. */
  def summariseUserSamples(perSample: Tensor1[Sample, Float32]): Tensor0[Float32] =
    perSample.mean

  def report(posterior: Tensor3[Chain, Draw, Sample, Float32]): Tensor0[Float32] =
    val flat = flattenPosterior(posterior)
    // Averaging over the user's Sample axis leaves one value per stacked draw. Handing
    // that to a routine expecting one value per user sample is the collision, and here
    // it is a type error rather than a name clash.
    summariseUserSamples(flat.mean(Axis[Sample]))
