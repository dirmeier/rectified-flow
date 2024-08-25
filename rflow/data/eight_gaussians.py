import numpy as np
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd


def eight_gaussians_fn():
    p_noise = tfd.Normal(0.0, 0.1)
    p_categories = tfd.Categorical(jnp.ones(8))
    centers = jnp.array(
        [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
    )

    def _fn(rng_key, n_samples):
        centers_key, noise_key = jr.split(rng_key)
        z = p_categories.sample(seed=centers_key, sample_shape=(n_samples,))
        noise = p_noise.sample(
            seed=noise_key,
            sample_shape=(
                n_samples,
                2,
            ),
        )
        data = centers[z] + noise
        return data

    return _fn
