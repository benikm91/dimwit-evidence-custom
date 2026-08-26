# Plain (NumPy) — verdict

**Category: `missed`**

`audio[i * HOP : i * HOP + WINDOW]` slices the first axis of whatever it is given. On a
`[batch, samples]` array that is the *batch*, so each "frame" is a stack of whole clips and
`len(audio)` is the batch size rather than the sample count.

The result is a 3-D array of finite numbers — `(1, 12, 3)` instead of `(5, 5, 3)` — so a
caller that only checks `ndim` sees nothing wrong
(`test_the_batched_call_runs_without_an_exception`). The mel projection survives the extra
axis because it contracts the window axis by position, and that position is still occupied.

`test_the_frame_axis_now_counts_clips_not_time_steps` names the tell: the middle axis of the
buggy output is the *sample count*, which no correct output could ever have. There is no
input size at which the buggy version coincides with the correct one — and for a batch
smaller than the window it does crash
(`test_a_batch_too_short_to_frame_crashes_instead`), which is the only regime where the
defect announces itself.
