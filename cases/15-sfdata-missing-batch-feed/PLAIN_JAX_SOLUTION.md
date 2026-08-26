# Plain JAX — verdict

**Category: `missed`**

JAX has no placeholders, so the one mechanism that caught this in TensorFlow is absent.
`jnp.reshape(x, (-1, FLAT))` maps both `(100, 100, 3)` and `(1, 100, 100, 3)` to
`(1, 30000)`, and `test_a_single_example_and_a_batch_of_one_are_indistinguishable_after_reshape`
asserts the two give *identical* results.

That identity is the trap: for batch size 1 the buggy and correct programs agree exactly, so
the bug is invisible until someone increases the batch size, at which point the model starts
training on a single example per step while the loop believes otherwise.

`test_the_label_side_broadcasts_a_missing_batch_axis_into_existence` covers the label half.
