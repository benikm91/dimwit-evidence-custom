# jaxtyping — verdict

**Category: `missed`**

`Float[Array, "out_h out_w"]` has two free variables and binding both to 84 satisfies it.
`test_an_annotation_could_only_forbid_squares_which_is_wrong` makes the deeper point: the
only annotation that would flag the buggy version is one asserting the two axes *differ*,
and that assertion is false — an 84x84 target is legitimate when the caller wants it.

There is no shape property that distinguishes "a square because the user chose one" from
"a square because the API could not express anything else".
