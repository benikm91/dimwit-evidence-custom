# 13 — `log_mel_spectrogram` had no batch dimension

**Source:** [openai/whisper#839](https://github.com/openai/whisper/pull/839)
— *"Support batch-dimension in log_mel_spectogram"*

## The defect

Whisper's feature extractor was written for a single audio waveform. Callers who wanted to
process several clips at once passed a `[batch, samples]` array, and the framing step — which
slices the *first* axis — sliced across clips instead of across time. The output has a
plausible shape and contains frames assembled from the wrong data.

The PR adds batch support explicitly. The alternative, which DimWit takes, is to keep the
function single-example and lift it.

## Why it is interesting

Same family as cases 09, 10 and 15 — a missing leading axis — but the failure is
*positional slicing inside a body* rather than a rank mismatch at a boundary. That matters
for the comparison: jaxtyping's boundary annotation catches the rank, but nothing in the
Python versions constrains what `audio[i : i + window]` slices.
