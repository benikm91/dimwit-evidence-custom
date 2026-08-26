# jaxtyping — verdict

**Category: `missed`, unconditionally**

Cases 01 and 05 were missed *because two axes happened to share a size*. This one is
missed even when the sizes differ, which makes it a qualitatively different limitation.

A softmax is shape-preserving whichever axis it reduces. `Float[Array, "batch cls"] ->
Float[Array, "batch cls"]` is a true and complete description of both the correct and the
incorrect program. There is no size fact left to check.
`test_both_satisfy_the_annotation_even_when_the_sizes_differ` uses `batch=3, cls=4` to make
that explicit.
