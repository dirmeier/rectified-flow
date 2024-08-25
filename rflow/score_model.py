import jax
from flax import linen as nn
from jax import numpy as jnp


def _timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
    """Sinusoidal embedding.

    From https://github.com/google-research/vdm/blob/main/model_vdm.py#L298C1-L323C13
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class ScoreModel(nn.Module):
    @staticmethod
    def time_embedding(times):
        times = _timestep_embedding(times, 128)
        times = nn.Sequential(
            [
                nn.Dense(256),
                nn.swish,
                nn.Dense(256),
                nn.swish,
            ]
        )(times)
        return times

    @nn.compact
    def __call__(self, inputs, times, **kwargs):
        # time embedding using sinusoidal embedding
        time_embedding = self.time_embedding(times)
        outputs = nn.Sequential(
            [
                nn.Dense(256),
                nn.swish,
                lambda x: x + time_embedding,
                nn.Dense(256),
                nn.swish,
                nn.Dense(256),
                nn.swish,
                nn.Dense(2),
            ]
        )(inputs)
        return outputs
