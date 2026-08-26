# Plain JAX — verdict

**Category: `run-time detection`**

A rare case where plain JAX beats plain NumPy. `jax.lax.dynamic_slice` requires the number
of start indices to match the operand rank, so `log_mel_single(BATCH)` raises
(`test_the_batched_call_fails_at_runtime_here`). Rewriting the same function with ordinary
Python slicing would be silent again, so the detection is an accident of which primitive the
author reached for.

`test_vmap_is_the_fix_and_it_is_one_word` shows the constructive answer, and
`test_but_nothing_required_the_author_to_use_it` shows its limit: the single-clip
function's signature says nothing about how it should be lifted, so the caller has to know.
