# 15 — feature and label batches fed without their batch dimension

**Source:** SFData program `s55237206` — the motivating example of
[*An Empirical Study on Tensor Shape Faults in Deep Learning Systems*](https://arxiv.org/abs/2106.02887)
(arXiv:2106.02887). Dataset: [github.com/tensfa/tensfa](https://github.com/tensfa/tensfa),
`SFData/StackOverflow/s55237206_*.py`. Derived from StackOverflow question 55237206.

```
ValueError: Cannot feed value of shape (100, 100, 3) for Tensor Placeholder:0,
            which has shape (1, 100, 100, 3)
```

The training loop iterates over examples and feeds `X_training[i]` and `Y_training[i]`,
each of which has lost the leading batch axis. The dataset's ground-truth patch inserts it:

```python
batch_x = np.expand_dims(batch_X, 0)
batch_Y = np.expand_dims(batch_Y, 0)
```

## Why it is in the dossier

It is the **representative** of the largest fault class in the published corpus. In SFData,
feature-input and label-output incompatibilities account for **65.8%** of the 146 crashing
tensor shape faults. Including it lets the paper connect its fifteen hand-picked cases to a
peer-reviewed prevalence figure instead of asserting one.

It is also the one case here that plain TensorFlow *did* catch — at run time, mid-training —
which is a useful reminder that the alternative to compile-time checking is not "no checking"
but "checking after the GPU has been busy for an hour".
