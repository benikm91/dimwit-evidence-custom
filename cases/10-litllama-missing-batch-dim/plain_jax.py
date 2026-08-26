"""Case 10 in JAX."""

import jax
import jax.numpy as jnp
import numpy as np

VOCAB, DIM = 7, 4
EMB = jnp.asarray(np.random.default_rng(3).normal(size=(VOCAB, DIM)), dtype=jnp.float32)
W_OUT = jnp.asarray(np.random.default_rng(4).normal(size=(DIM, VOCAB)), dtype=jnp.float32)


def model(idx):
    x = EMB[idx]
    x = x - x.mean(axis=0, keepdims=True)
    return x @ W_OUT


IDX = jnp.array([1, 3, 5, 2])


# --------------------------------------------------------------------------- tests

def test_both_ranks_trace_successfully():
    assert model(IDX).shape == (4, VOCAB)
    assert model(IDX.reshape(1, -1)).shape == (1, 4, VOCAB)


def test_jit_compiles_the_unbatched_call():
    assert jax.jit(model)(IDX).shape == (4, VOCAB)


def test_vmap_is_the_thing_that_would_have_forced_the_question():
    """Mapping the single-example model makes the batch axis explicit and external."""
    batched = jax.vmap(model)
    out = batched(IDX.reshape(1, -1))
    assert out.shape == (1, 4, VOCAB)
    # and note it is NOT the same as calling the model on the 2-D input directly,
    # because the centring step now sees one example at a time
    assert not np.allclose(np.asarray(out), np.asarray(model(IDX.reshape(1, -1))))
