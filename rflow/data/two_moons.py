import numpy as np
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


def two_moons_fn():
    p_noise = tfd.Normal(0.0, 0.1)

    def _fn(rng_key, n_samples):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out

        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 1

        data = jnp.vstack(
            [
                np.append(outer_circ_x, inner_circ_x),
                np.append(outer_circ_y, inner_circ_y),
            ]
        ).T
        data = data + p_noise.sample(seed=rng_key, sample_shape=data.shape)
        data = data * 0.35
        data = data - jnp.mean(data, axis=0, keepdims=True)
        return data

    return _fn
