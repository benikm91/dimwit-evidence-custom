# Plain (TensorFlow / NumPy) — verdict

**Category: `run-time detection` on the feature path, `missed` on the label path**

The feature path is the one case in this dossier where the original stack does its job.
TensorFlow's placeholder carries a declared shape, and feeding `(100, 100, 3)` into a
`(1, 100, 100, 3)` placeholder raises the error quoted in the README
(`test_the_buggy_feed_raises_the_upstream_error`). That is a run-time failure, mid-training,
after the graph has been built and the session started — but it is a failure.

Two qualifications matter for the paper:

* `test_the_reshape_path_would_have_been_silent` — the check came from the *declared*
  placeholder shape, not from the arithmetic. The model's first op is
  `tf.reshape(X, [-1, 30000])`, which accepts rank 3 and rank 4 alike. Declare the
  placeholder `(None, 100, 100, 3)`, as most people do, and nothing raises.
* `test_the_label_path_is_silently_wrong_rather_than_loud` — a one-hot label of shape `(2,)`
  multiplied against `(1, 2)` logits broadcasts a batch axis into existence. That half of the
  same bug is silent even in TensorFlow.

So the honest summary is: a run-time check fired here because the author happened to
hard-code a batch size of 1 in the placeholder.
