# 02 — MSE between `[n, 1]` predictions and `[n]` targets

**Source:** [pytorch/pytorch#16045](https://github.com/pytorch/pytorch/issues/16045)
— *"fail to throw error when computing loss between tensors with shapes [n, 1] and [n]"*

## The defect

`nn.Linear(2, 1)` returns `[n, 1]`. The targets are `[n]`. `nn.MSELoss` subtracts them,
broadcasting to `[n, n]` — the full matrix of pairwise residuals — and averages that.

The loss decreases. The model trains. The weights are wrong.

The reporter noticed because a two-variable linear regression, a convex problem with a
closed-form solution, refused to reach its optimum. PyTorch's resolution was a
`UserWarning`, not an error: implicit broadcasting is too load-bearing to remove.

## Why it is interesting

This is the case where **named axes alone are not enough**. Haliax broadcasts by name
implicitly, so subtracting a `{batch}` array from a `{batch, out}` array is just as quiet
there. What prevents it in DimWit is the separate elementwise operators: `-` demands
identical shapes, `-!` broadcasts and has to be written down.
