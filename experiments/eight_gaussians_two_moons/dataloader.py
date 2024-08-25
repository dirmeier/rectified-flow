import dataclasses

from jax import random as jr
from tensorflow_probability.substrates import jax as tfp

from rflow import data

tfd = tfp.distributions


def get_data_loaders(
    rng_key,
    datasets=["two_moons", "eight_gaussians"],
):
    data_fns = [getattr(data, dataset + "_fn")() for dataset in datasets]

    @dataclasses.dataclass
    class _DataLoader:
        rng_key: jr.PRNGKey

        def __iter__(self):
            while True:
                yield self()

        def __call__(self, batch_size=256):
            *batch_keys, self.rng_key = jr.split(self.rng_key, len(data_fns) + 1)
            batch = [data_fn(batch_key, batch_size) for batch_key, data_fn in zip(batch_keys, data_fns)]
            return batch

    return _DataLoader(rng_key)
