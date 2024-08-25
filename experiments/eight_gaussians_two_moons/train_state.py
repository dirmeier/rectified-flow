import optax
from flax.training.train_state import TrainState


def new_train_state(rng_key, model, init_batch):
    variables = model.init(
        {"params": rng_key, "sample": rng_key},
        method="loss",
        inputs=init_batch,
        is_training=False,
    )
    tx = optax.adamw(0.0003)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )
