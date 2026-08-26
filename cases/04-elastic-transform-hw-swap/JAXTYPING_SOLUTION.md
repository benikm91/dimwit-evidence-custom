# jaxtyping — verdict

**Category: `missed`**

Every array annotation in `jaxtyping_case.py` is truthful and satisfied. The divisors are
declared `height: int` and `width: int` — the same Python type — so
`test_int_arguments_carry_no_axis_identity` shows that even *swapping the two arguments at
the call site* type-checks.

jaxtyping's vocabulary covers array axes. The defect lives in the scalars extracted from
those axes, one step outside that vocabulary.
