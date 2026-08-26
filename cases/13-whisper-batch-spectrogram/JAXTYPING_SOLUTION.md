# jaxtyping — verdict

**Category: `run-time detection`**

`Float[Array, "samples"]` fixes the rank at 1, so the batched call raises. One of the four
rank cases (09, 10, 13, 15) where jaxtyping does its job well.

Note also `test_vmap_composes_with_the_annotation`: `jax.vmap` of a `@jaxtyped` function
still works, so the annotation records the single-clip contract *and* the lifted use is
unaffected. That is the closest any of the Python tools gets to DimWit's arrangement, and
the paper should credit it. The remaining differences are that the check is a run-time one
and that it constrains rank rather than identity — nothing stops the body from slicing the
wrong axis of a rank-1 array, there just is not another one.
