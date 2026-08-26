//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 15 (buggy) — the SFData motivating example, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Transliteration of
  *   batch_X, batch_Y = X_training[i], Y_training[i]
  *   sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})
  * A single example is handed to the batched training step.
  *
  * Expected compiler errors: `Tensor3[Height, Width, Channel, Float32]` where
  * `Tensor4[Batch, Height, Width, Channel, Float32]` is required, and likewise
  * `Tensor1[Class, Float32]` where `Tensor2[Batch, Class, Float32]` is required.
  */
object Case15Buggy:

  import dimwit.*

  trait Batch derives Label
  trait Height derives Label
  trait Width derives Label
  trait Channel derives Label
  trait Class derives Label

  def batchLoss(
      images: Tensor4[Batch, Height, Width, Channel, Float32],
      labels: Tensor2[Batch, Class, Float32],
      w: Tensor2[Height |*| Width |*| Channel, Class, Float32],
      b: Tensor1[Class, Float32]
  ): Tensor0[Float32] =
    zipvmap(Axis[Batch])(images, labels)(
      (img: Tensor3[Height, Width, Channel, Float32], lbl: Tensor1[Class, Float32]) =>
        -(lbl * (img.flatten.dot(Axis[Height |*| Width |*| Channel])(w) + b).abs.log).sum
    ).mean

  def trainStep(
      images: Tensor4[Batch, Height, Width, Channel, Float32],
      labels: Tensor2[Batch, Class, Float32],
      w: Tensor2[Height |*| Width |*| Channel, Class, Float32],
      b: Tensor1[Class, Float32],
      i: Int
  ): Tensor0[Float32] =
    val batchX = images.slice(Axis[Batch].at(i)) // Tensor3: the batch axis is gone
    val batchY = labels.slice(Axis[Batch].at(i)) // Tensor1: likewise
    batchLoss(batchX, batchY, w, b)
