import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax import random as jr
from scipy import integrate
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class RectifiedFlow(nn.Module):
    score_model: nn.Module

    def __call__(self, method="loss", **kwargs):
        return getattr(self, method)(**kwargs)

    def loss(self, inputs, is_training):
        y1, y0 = inputs
        rng_key = self.make_rng("sample")
        time_key, rng_key = jr.split(rng_key)
        times = jr.uniform(time_key, (y0.shape[0],), minval=1e-3)
        inputs = (
            times.reshape((y1.shape[0], 1)) * y1
            + (1.0 - times.reshape((y0.shape[0], 1))) * y0
        )
        ret = self.score_model(inputs, times, is_training=is_training)
        target = y1 - y0
        loss = jnp.sum(jnp.square(target - ret), axis=range(1, y0.ndim))
        return loss

    def sample(self, inputs, is_training=False, eps=1e-8):
        def ode_func(t, yt):
            yt = yt.reshape(-1, *inputs.shape[1:])
            t = np.full((yt.shape[0],), t)
            ret = self.score_model(yt, t, is_training=is_training)
            return ret.reshape(-1)

        ret = integrate.solve_ivp(
            ode_func,
            (1.0, eps),
            np.asarray(inputs).reshape(-1),
            rtol=1e-5,
            atol=1e-5,
            method="RK45",
        )

        ret = ret.y[:, -1].reshape(inputs.shape)
        return ret
