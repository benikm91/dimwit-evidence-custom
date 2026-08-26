//> using scala 3.8.1
//> using repository ivy2Local
//> using dep ch.contrafactus::dimwit-core:0.2-SNAPSHOT

/** Case 15 (fixed) — the SFData motivating example, DimWit.
  *
  * The classifier is written for ONE image and ONE label. Training over a batch is `vmap`,
  * so there is no batch axis to feed, forget or invent.
  */
object Case15Fixed:

  import dimwit.*

  trait Batch derives Label
  trait Height derives Label
  trait Width derives Label
  trait Channel derives Label
  trait Pixel derives Label
  trait Class derives Label

  /** Logits for one image. */
  def logits(
      image: Tensor3[Height, Width, Channel, Float32],
      w: Tensor2[Height |*| Width |*| Channel, Class, Float32],
      b: Tensor1[Class, Float32]
  ): Tensor1[Class, Float32] =
    image.flatten.dot(Axis[Height |*| Width |*| Channel])(w) + b

  /** Cross entropy for one image and its one-hot label. */
  def loss(
      image: Tensor3[Height, Width, Channel, Float32],
      label: Tensor1[Class, Float32],
      w: Tensor2[Height |*| Width |*| Channel, Class, Float32],
      b: Tensor1[Class, Float32]
  ): Tensor0[Float32] =
    -(label * logits(image, w, b).abs.log).sum

  /** The training objective over a batch: the single-example loss, lifted. */
  def batchLoss(
      images: Tensor4[Batch, Height, Width, Channel, Float32],
      labels: Tensor2[Batch, Class, Float32],
      w: Tensor2[Height |*| Width |*| Channel, Class, Float32],
      b: Tensor1[Class, Float32]
  ): Tensor0[Float32] =
    zipvmap(Axis[Batch])(images, labels)(
      (img: Tensor3[Height, Width, Channel, Float32], lbl: Tensor1[Class, Float32]) => loss(img, lbl, w, b)
    ).mean

  @main def case15Check(): Unit =
    dimwit.initialize()

    // 10x10x3 rather than 100x100x3 so the check runs quickly; the shapes are the point
    val w = Tensor(Shape(Axis[Height |*| Width |*| Channel] -> 300, Axis[Class] -> 2)).fill(0.01f)
    val b = Tensor1(Axis[Class]).fromArray(Array(0f, 0f))

    val images = Tensor(Shape(Axis[Batch] -> 4, Axis[Height] -> 10, Axis[Width] -> 10, Axis[Channel] -> 3)).fill(0.5f)
    val labels = Tensor(Shape(Axis[Batch] -> 4, Axis[Class] -> 2))
      .fromArray(Array(1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f))

    val l = batchLoss(images, labels, w, b)
    assert(l.item.isFinite, s"loss must be finite, got ${l.item}")

    // a single example goes through the single-example entry point, with no axis to add
    val one = images.slice(Axis[Batch].at(0))
    assert(logits(one, w, b).shape(Axis[Class]) == 2)
    println(s"case15 ok: per-example loss ${l.item}, batching by zipvmap")
