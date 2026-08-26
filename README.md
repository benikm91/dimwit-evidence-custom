# dimwit-evidence-custom

Fifteen real tensor-shape defects, taken from the issue trackers of well-known repositories
and from a published fault corpus, each reconstructed four times — plain NumPy/PyTorch,
plain JAX, JAX + jaxtyping, and DimWit — so that the claim *"names as types catch this at
compile time"* can be checked rather than asserted.

Built for the evaluation section of the DimWit paper, in response to the review note:

> 10–15 reale Shape-Bugs aus GitHub-Issues bekannter Repos nachbauen; Tabelle: welche fängt
> DimWit zur Compile-Zeit vs. Haliax / jaxtyping / plain JAX.

Haliax is out of scope here, as requested.

## Running it

```sh
./run_all.sh            # pytest over all reconstructions, then the DimWit compile checks
uv run pytest -q        # just the Python side
./scripts/check_scala.sh -v   # just the Scala side, with compiler output
```

Requirements: [uv](https://github.com/astral-sh/uv) and
[scala-cli](https://scala-cli.virtuslab.org/). The Scala sources resolve
`ch.contrafactus::dimwit-core:0.2-SNAPSHOT` from the local ivy cache, so run `sbt publishLocal`
in `../dimwit` first if it is not already there.

One test is skipped by default: `cases/02/plain.py::test_torch_only_warns` needs PyTorch,
which is not in the lock file. It was verified separately against torch 2.5.1 — see
`RESULTS.md`.

## What is in each case folder

| file | contents |
|---|---|
| `README.md` | the defect, a link to the upstream issue and fix, and why the case is here |
| `plain.py` | buggy and fixed reconstruction in NumPy (or PyTorch where the semantics matter), with tests that assert the *erroneous* behaviour as well as the correct one |
| `plain_jax.py` | the same in JAX |
| `jaxtyping_case.py` | the same annotated with jaxtyping + beartype |
| `Buggy.scala` | the DimWit transliteration of the mistake — normally rejected by the compiler |
| `Fixed.scala` | the DimWit program that is correct, with a `@main` check that runs it |
| `PLAIN_SOLUTION.md`, `PLAIN_JAX_SOLUTION.md`, `JAXTYPING_SOLUTION.md`, `DIMWIT_SOLUTION.md` | one verdict each, with the reasoning and the honest limits |

## Categories

Every verdict is one of three:

* **`compile-time detection`** — the compiler rejects the program. Nothing runs.
* **`run-time detection`** — the program starts and fails with an error on the path that
  contains the defect.
* **`missed`** — the program runs to completion and produces a result of the right shape and
  the wrong values. Only inspecting the numbers reveals it.

## Results

Verified by `run_all.sh`: **162 Python tests pass**, all 15 `Fixed.scala` compile, 13 of 15
`Buggy.scala` are rejected at compile time, and the 2 that are not are recorded as `missed`.

| # | case | source | plain | plain JAX | jaxtyping | **DimWit** |
|---|---|---|---|---|---|---|
| 01 | attention mask on the query axis | [transformers#23974](https://github.com/huggingface/transformers/issues/23974) | missed | missed | missed¹ | **compile** |
| 02 | MSE over `[n,1]` vs `[n]` | [pytorch#16045](https://github.com/pytorch/pytorch/issues/16045) | missed (warns) | missed | missed² | **compile** |
| 03 | `cross` on the first length-3 axis | [keras#23219](https://github.com/keras-team/keras/pull/23219) | missed | missed | missed | **compile** |
| 04 | displacement normalised by the wrong side | [torchvision#9299](https://github.com/pytorch/vision/issues/9299) | missed | missed | missed | **compile**³ |
| 05 | NCHW height/width swapped | [ultralytics#23126](https://github.com/ultralytics/ultralytics/issues/23126) | missed | missed | missed⁴ | **compile** |
| 06 | softmax over the wrong axis | [triton#11406](https://github.com/triton-lang/triton/pull/11409) | missed | missed | missed | **missed** |
| 07 | stacked dim name collided | [arviz#1647](https://github.com/arviz-devs/arviz/pull/1647) | missed | missed | missed | **compile** |
| 08 | reshape with pre-transpose sizes | [pymc#7178](https://github.com/pymc-devs/pymc/issues/7178) | missed | missed | run-time⁵ | **compile** |
| 09 | single G-vector broadcast | [pyscf#2961](https://github.com/pyscf/pyscf/issues/2961) | missed | missed | run-time | **compile** |
| 10 | model called without a batch axis | [lit-llama#166](https://github.com/Lightning-AI/lit-llama/pull/166) | missed | missed | run-time | **compile** |
| 11 | NCHW frames into an NHWC pipeline | [vllm#51076](https://github.com/vllm-project/vllm/pull/51076) | missed | missed | missed | **compile** |
| 12 | batched contraction over the wrong axis | [mlx#4125](https://github.com/ml-explore/mlx/pull/4125) | missed⁵ | missed⁵ | missed⁵ | **compile** |
| 13 | spectrogram with no batch dimension | [whisper#839](https://github.com/openai/whisper/pull/839) | missed | run-time | run-time | **compile** |
| 14 | square-only `screen_size` | [Gymnasium#1312](https://github.com/Farama-Foundation/Gymnasium/pull/1312) | missed | missed | missed | **missed** |
| 15 | example fed without its batch axis | SFData `s55237206` | run-time⁶ | missed | run-time | **compile** |

¹ caught for cross-attention, where the query and key lengths differ; Pix2Struct is self-attention.
² caught if the loss is annotated `"batch"` on both arguments; missed with the signature the buggy model actually had.
³ conditional on keeping the extent wrapped as `AxisExtent[Width]` rather than unwrapping it to `Int`.
⁴ jaxtyping binds axis names only from *array* annotations, so `height: int` constrains nothing. See case 05.
⁵ caught only when the two candidate axes have different extents.
⁶ the feature path raised; the label path broadcast silently.

**Totals.** DimWit: 13 compile-time, 2 missed, 0 run-time.
jaxtyping: 4 unconditional run-time, 5 conditional, 6 missed.
plain JAX: 1 run-time, 14 missed. plain: 1 run-time, 14 missed (one with a warning).

## The two DimWit misses are the point of the exercise

`06` and `14` are included deliberately and their `Buggy.scala` files compile.

* **06 — softmax over the wrong axis.** A shape-preserving reduction has the same type
  whichever axis it reduces, so no shape-indexed type system can prefer one. This rules out
  a whole family: `mean`, `std`, `cumsum`, layer norm. DimWit turns `dim=0` into
  `Axis[Batch]`, which helps review and removes action-at-a-distance when axes are
  reordered — but it is not detection.
* **14 — one integer used for two axes.** `Shape(Axis[Height] -> s, Axis[Width] -> s)` must
  be well-typed, because square tensors are legitimate. DimWit constrains how values combine,
  not which values the program should have chosen.

## Corrections the experiment produced

Building the cases changed three verdicts from the initial estimate, and the changes are
worth carrying into the paper:

1. **Case 14 was estimated "partly prevented"; it is `missed`.** `Buggy.scala` compiles.
2. **Case 05 under jaxtyping was estimated "run-time detection"; it is `missed` as written.**
   jaxtyping does not bind names from scalar `int` parameters, so the return annotation
   constrains nothing. It only fires if the layout is carried by an array *and* the model
   input is rectangular.
3. **Case 04's DimWit result is conditional**, not unqualified: `shape(Axis[Height]): Int`
   unwraps the extent and gives the guarantee up.

## Suggested next step

Case 15 is the representative of SFData's largest class — feature-input and label-output
mismatches are 65.8% of its 146 crashing shape faults. Classifying all 146 patches by the
DimWit mechanism that would have prevented them converts this hand-picked table into a
dataset-scale result over peer-reviewed data. Dataset:
[github.com/tensfa/tensfa](https://github.com/tensfa/tensfa),
paper: [arXiv:2106.02887](https://arxiv.org/abs/2106.02887).
