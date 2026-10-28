"""
Gaussian-process critic for structural policy gradient (SPG / RSPG).

The critic predicts the value of *every* individual state at once (the GP's outputs)
from a handful of aggregate features (the GP's inputs). Rollouts can then be truncated
after H periods, and the GP values the remaining future instead of simulating it.

- Inputs: aggregate features x = (z, K), optionally the time left in the episode (a
  finite-horizon, time-aware critic), and optionally lagged (z, K).
  For the endogenous model, (z, K) is a one-to-one map of the observed prices (r, wage)
  because average income is constant.
- Outputs: one value per individual state (n_states outputs), sharing one kernel.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax.scipy.linalg import cho_solve

JITTER = 1e-4


@struct.dataclass
class GPHyper:
    log_ell: jax.Array  # (D,) one length scale per (standardized) input
    log_sf: jax.Array  # () signal std
    log_sn: jax.Array  # () noise std


@struct.dataclass
class GPState:
    hyp: GPHyper
    X: jax.Array  # (n, D) standardized training inputs
    alpha: jax.Array  # (n, J) K^-1 Y (standardized outputs)
    x_mean: jax.Array  # (D,)
    x_std: jax.Array  # (D,)
    y_mean: jax.Array  # (J,)
    y_std: jax.Array  # (J,)


# --- features ---
def aggregate_features(env, aggregate_s) -> jax.Array:
    """
    (z, K) for a (batch of) endogenous aggregate state(s). K is average wealth.
    Works on a single state (returns (2,)) or a batch (returns (E, 2)).
    """
    nw, ny = env.params.num_states
    mu = aggregate_s.mu.reshape(*aggregate_s.mu.shape[:-1], nw, ny)
    K = jnp.sum(mu.sum(-1) * env.params.discrete_states[0], axis=-1)
    z = jnp.asarray(aggregate_s.z, dtype=K.dtype)
    return jnp.stack([z, K], axis=-1)


def init_feature_history(features: jax.Array, n_lags: int) -> jax.Array:
    """(E, 2) -> (E, n_lags + 1, 2), newest first. Lags are filled with the current value at reset."""
    return jnp.repeat(features[:, None, :], n_lags + 1, axis=1)


def push_feature_history(history: jax.Array, features: jax.Array) -> jax.Array:
    """Prepend the newest features and drop the oldest lag."""
    return jnp.concatenate([features[:, None, :], history[:, :-1, :]], axis=1)


def gp_inputs(history: jax.Array, time_left: jax.Array | None = None) -> jax.Array:
    """
    (E, n_lags + 1, 2) -> [z_t, K_t, z_{t-1}, K_{t-1}, ...], or, with the time left in the
    episode, [z_t, K_t, T - t, z_{t-1}, K_{t-1}, ...].
    """
    x = history.reshape(history.shape[0], -1)
    if time_left is None:
        return x
    return jnp.concatenate(
        [x[:, :2], jnp.asarray(time_left, x.dtype)[:, None], x[:, 2:]], axis=1
    )


def n_base_inputs(time_input: bool) -> int:
    """Inputs that are not lags: (z, K), plus the time left when the critic is time-aware."""
    return 3 if time_input else 2


# --- GP ---
def init_hyper(n_inputs: int, n_base: int = 2, lag_ell: float = 10.0) -> GPHyper:
    """
    Unit length scales for the current features; long length scales for the lags, so that
    the fit starts from the no-lag model (Adam got stuck in a worse optimum otherwise).
    """
    log_ell = jnp.concatenate(
        [jnp.zeros(n_base), jnp.full(n_inputs - n_base, jnp.log(lag_ell))]
    )
    return GPHyper(
        log_ell=log_ell.astype(jnp.float32),
        log_sf=jnp.array(0.0, dtype=jnp.float32),
        log_sn=jnp.array(jnp.log(0.1), dtype=jnp.float32),
    )


def empty_gp(n: int, n_inputs: int, n_outputs: int) -> GPState:
    """A GP with zero weights: predicts 0 everywhere. Used before the first fit."""
    return GPState(
        hyp=init_hyper(n_inputs),
        X=jnp.zeros((n, n_inputs)),
        alpha=jnp.zeros((n, n_outputs)),
        x_mean=jnp.zeros(n_inputs),
        x_std=jnp.ones(n_inputs),
        y_mean=jnp.zeros(n_outputs),
        y_std=jnp.ones(n_outputs),
    )


def kernel(X1, X2, hyp: GPHyper):
    diff = (X1[:, None, :] - X2[None, :, :]) / jnp.exp(hyp.log_ell)
    return jnp.exp(2 * hyp.log_sf) * jnp.exp(-0.5 * jnp.sum(diff**2, -1))


def _gram(X, hyp: GPHyper):
    n = X.shape[0]
    return kernel(X, X, hyp) + (jnp.exp(2 * hyp.log_sn) + JITTER) * jnp.eye(n)


def neg_log_marginal_likelihood(hyp: GPHyper, X, Y):
    """Per-point negative log marginal likelihood, averaged over the J outputs (Y standardized)."""
    n = X.shape[0]
    L = jnp.linalg.cholesky(_gram(X, hyp))
    alpha = cho_solve((L, True), Y)
    return (
        0.5 * jnp.sum(Y * alpha) / Y.shape[1] + jnp.sum(jnp.log(jnp.diag(L)))
    ) / n


def _standardize(X_raw, Y_raw):
    x_mean = X_raw.mean(0)
    x_std = X_raw.std(0)
    x_std = jnp.where(x_std > 1e-12, x_std, 1.0)
    y_mean = Y_raw.mean(0)
    y_std = jnp.maximum(Y_raw.std(0), 1e-8 + 1e-6 * jnp.abs(y_mean))
    return x_mean, x_std, y_mean, y_std


@partial(jax.jit, static_argnames=("n_steps",))
def fit_gp(X_raw, Y_raw, hyp_init: GPHyper, n_steps: int, lr: float = 0.05):
    """
    Fit hyperparameters by Adam on the negative log marginal likelihood (warm-started from
    `hyp_init`), then solve for the weights. Returns (GPState, final NLML).
    """
    x_mean, x_std, y_mean, y_std = _standardize(X_raw, Y_raw)
    X = (X_raw - x_mean) / x_std
    Y = (Y_raw - y_mean) / y_std

    tx = optax.adam(lr)
    loss_and_grad = jax.value_and_grad(neg_log_marginal_likelihood)

    def _step(carry, _):
        hyp, opt_state = carry
        loss, grads = loss_and_grad(hyp, X, Y)
        updates, opt_state = tx.update(grads, opt_state)
        hyp = optax.apply_updates(hyp, updates)
        return (hyp, opt_state), loss

    (hyp, _), _ = jax.lax.scan(
        _step, (hyp_init, tx.init(hyp_init)), None, length=n_steps
    )

    L = jnp.linalg.cholesky(_gram(X, hyp))
    alpha = cho_solve((L, True), Y)
    nll = neg_log_marginal_likelihood(hyp, X, Y)
    return GPState(hyp, X, alpha, x_mean, x_std, y_mean, y_std), nll


def gp_predict(gp: GPState, x_raw: jax.Array) -> jax.Array:
    """(m, D) raw inputs -> (m, J) predicted values."""
    x = (x_raw - gp.x_mean) / gp.x_std
    return gp.y_mean + gp.y_std * (kernel(x, gp.X, gp.hyp) @ gp.alpha)


def subsample(key, X, Y, n: int):
    """Uniform subsample of n training pairs (without replacement when possible)."""
    N = X.shape[0]
    idx = jax.random.choice(key, N, (n,), replace=N < n)
    return X[idx], Y[idx]
