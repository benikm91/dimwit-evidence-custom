# Plain JAX — verdict

**Category: `missed`**

`jax.eval_shape` cheerfully confirms an `(84, 84)` output — which *is* the bug
(`test_shape_inference_confirms_a_square_output_which_is_exactly_the_bug`).

Worth stating plainly in the paper: shape inference verifies that a program computes the
shape it claims. It cannot verify that the claim was the right one. Every tool in this
comparison, DimWit included, is a consistency checker, not an oracle.
