import jax
import numpy as np
from absl import app, logging
from dataloader import get_data_loaders
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from experiments.eight_gaussians_two_moons.train_state import new_train_state
from rflow import RectifiedFlow, ScoreModel


def get_model():
    model = RectifiedFlow(ScoreModel())
    return model


@jax.jit
def step_fn(step_key, state, batch):
    def loss_fn(params, rng):
        sample_key, dropout_key = jr.split(rng)
        ll = state.apply_fn(
            variables={"params": params},
            rngs={"sample": sample_key},
            method="loss",
            inputs=batch,
            is_training=True,
        )
        ll = jnp.mean(ll)
        return ll

    loss, grads = jax.value_and_grad(loss_fn)(state.params, step_key)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


def train(rng_key, data_loader):
    state_key, rng_key = jr.split(rng_key)

    model = get_model()
    state = new_train_state(state_key, model, next(iter(data_loader)))
    step_key, rng_key = jr.split(rng_key)

    for step in range(1, 10001):
        train_key, sample_key = jr.split(jr.fold_in(step_key, step))
        loss, state = step_fn(train_key, state, data_loader())
        if step == 1 or step % 250 == 0:
            logging.info(f"loss at epoch {step}: {loss}")
        if step == 1 or step % 1_000 == 0:
            sample(sample_key, step, data_loader(1_000), state)


def _sample(sample_key, state, batch):
    return state.apply_fn(
        variables={"params": state.params}, rngs={"sample": sample_key}, method="sample", inputs=batch[0]
    )


def sample(sample_key, step, batch, state):
    samples = _sample(sample_key, state, batch)
    samples = np.array(samples)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(batch[0][:, 0], batch[0][:, 1], marker=".", color="black", alpha=0.5)
    ax.scatter(samples[:, 0], samples[:, 1], marker=".", color="blue", alpha=0.5)
    ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
    ax.grid(True)
    ax.set_title(f"{step} training steps")
    fig.savefig(f"./figures/samples-{step}.png")
    plt.close()


def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)
    data_loader = get_data_loaders(rng_key=jr.PRNGKey(0))
    train(jr.PRNGKey(2), data_loader)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
