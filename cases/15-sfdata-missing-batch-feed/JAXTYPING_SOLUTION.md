# jaxtyping — verdict

**Category: `run-time detection`, on both paths**

The strongest jaxtyping showing in the dossier. `Float[Array, "batch h w c"]` rejects the
rank-3 feature, and — unlike TensorFlow — `Float[Array, "batch cls"]` on the loss also
rejects the rank-1 label, closing the half of the bug that broadcast silently upstream.

This is the fourth rank case (with 09, 10 and 13) and it confirms the pattern worth stating
once in the paper: **jaxtyping is reliable on rank, unreliable on identity.** Ranks cannot
coincide by accident; extents routinely do.

The gap that remains is the phase. These checks fire when the annotated function is first
called with real data — in the SFData program, inside the training loop.
