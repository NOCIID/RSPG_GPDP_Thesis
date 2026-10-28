"""
GPDP backward pass against a fixed population: the household's best response when it does NOT know
future aggregate shocks (build step 2 in FINDINGS.md).

    V_t(x) = max_a  r(x, a) + gamma * E_z' E_x'[ V_{t+1}(x'; z', K_{t+1}) | x, a ]

- "Stations": economies simulated forward under the population's policy. V_t is computed exactly at
  the stations, period by period, backward.
- K_{t+1} comes from the station's own push-forward (it does not depend on tomorrow's shock z').
- z' = rho z + nu eps is averaged over Gauss-Hermite points eps *before* the max, so the choice does
  not use the realized future shock.
- V_{t+1} at (z', K_{t+1}) comes from a GP fitted on the stations of period t+1 (gp_critic.py: inputs
  (z, K), one output per individual state), except after the last decision period, where it is the
  terminal reward, exact for any z'.
"""

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.linalg import cho_solve

from mfax.algos.hsm.utils import gpdp_continuous as gc
from mfax.algos.hsm.utils.gp_critic import (
    JITTER,
    kernel,
    GPHyper,
    GPState,
    _gram,
    _standardize,
    aggregate_features,
    gp_predict,
    init_hyper,
    neg_log_marginal_likelihood,
)

# --- bounds on the GP hyperparameters (standardized inputs and outputs) ---
# The backward pass's targets are exact DP values, a smooth and nearly noiseless function of (z, K).
# Unbounded, the marginal likelihood keeps shrinking the noise and growing the signal and length
# scales until the kernel matrix is singular in float32 and the fit returns NaN (the critic's noisy
# targets never triggered this). Noise of at least 1% of the outputs' spread keeps it well conditioned.
LOG_SN_BOUNDS = (float(np.log(1e-2)), 0.0)
LOG_SF_BOUNDS = (float(np.log(0.1)), float(np.log(10.0)))
LOG_ELL_BOUNDS = (float(np.log(0.05)), float(np.log(100.0)))


def _clip_hyper(hyp: GPHyper) -> GPHyper:
    return GPHyper(
        log_ell=jnp.clip(hyp.log_ell, *LOG_ELL_BOUNDS),
        log_sf=jnp.clip(hyp.log_sf, *LOG_SF_BOUNDS),
        log_sn=jnp.clip(hyp.log_sn, *LOG_SN_BOUNDS),
    )


@partial(jax.jit, static_argnames=("n_steps",))
def fit_gp_bounded(X_raw, Y_raw, hyp_init: GPHyper, n_steps: int, lr: float = 0.05):
    """
    gp_critic.fit_gp with the hyperparameters kept inside the bounds above (projected Adam), and the
    starting hyperparameters kept if the fit still produces a non-finite value.
    Returns (GPState, final NLML).
    """
    x_mean, x_std, y_mean, y_std = _standardize(X_raw, Y_raw)
    X = (X_raw - x_mean) / x_std
    Y = (Y_raw - y_mean) / y_std
    hyp_init = _clip_hyper(hyp_init)

    tx = optax.adam(lr)
    loss_and_grad = jax.value_and_grad(neg_log_marginal_likelihood)

    def _step(carry, _):
        hyp, opt_state = carry
        _, grads = loss_and_grad(hyp, X, Y)
        updates, opt_state = tx.update(grads, opt_state)
        return (_clip_hyper(optax.apply_updates(hyp, updates)), opt_state), None

    (hyp, _), _ = jax.lax.scan(
        _step, (hyp_init, tx.init(hyp_init)), None, length=n_steps
    )
    finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(hyp)])
    )
    hyp = jax.tree.map(lambda new, old: jnp.where(finite, new, old), hyp, hyp_init)

    L = jnp.linalg.cholesky(_gram(X, hyp))
    alpha = cho_solve((L, True), Y)
    nll = neg_log_marginal_likelihood(hyp, X, Y)
    return GPState(hyp, X, alpha, x_mean, x_std, y_mean, y_std), nll


def shock_nodes(n: int = 5):
    """Gauss-Hermite points and probabilities for a standard normal shock."""
    eps, w = np.polynomial.hermite_e.hermegauss(n)
    return jnp.asarray(eps, jnp.float32), jnp.asarray(w / w.sum(), jnp.float32)


def _with_shock(env, aggregate_s, z):
    """The same aggregate state with another shock z: same mu, prices recomputed."""
    interest_rate, wage = env._prices(aggregate_s.mu, z)
    return aggregate_s.replace(z=z, interest_rate=interest_rate, wage=wage)


def continuation(env, aggregate_s, next_aggregate_s, gp_next, eps, probs, terminal_next, gamma=None):
    """
    W(x') = E_z'[ V_{t+1}(x'; z', K_{t+1}) ] for one economy: tomorrow's value of every individual
    state, averaged over tomorrow's shock. Only mu_{t+1} is used from next_aggregate_s; its realized
    shock is ignored.
    """
    z_next = env.params.rho * aggregate_s.z + env.params.nu * eps  # (n_nodes,)
    if terminal_next:
        # --- the next period is the last one: its value is the terminal reward, for any share ---
        values = jax.vmap(
            lambda z: gc.reward(
                env,
                jnp.full((env.n_states,), 0.5),
                _with_shock(env, next_aggregate_s, z),
                True,
            )
        )(z_next)
    else:
        K_next = aggregate_features(env, next_aggregate_s)[1]
        x = jnp.stack([z_next, jnp.full_like(z_next, K_next)], axis=-1)
        if gp_next.X.shape[-1] == 3:
            # --- a single value GP for all periods: inputs (z, K, T - t), predicts log c_bar ---
            time_left = env.params.max_steps_in_episode - next_aggregate_s.time
            x = jnp.concatenate([x, jnp.full_like(z_next, time_left)[:, None]], axis=-1)
            values = log_ce_to_value(gp_predict(gp_next, x), time_left, gamma, env.params.sigma)
        else:
            values = gp_predict(gp_next, x)  # (n_nodes, n_states)
    return probs @ values


@lru_cache(maxsize=None)
def make_lookahead(env, gamma, n_nodes: int = 5, n_iter: int = 30, grid_only: bool = False):
    """
    Jitted one-period best response for a batch of economies (leading axis E):
    (aggregate_s, next_aggregate_s[, gp_next]) -> (shares (E, n_states), values (E, n_states)).
    Returns two functions: with a GP continuation, and with the exact terminal continuation.
    grid_only=True chooses among the grid actions only (exact argmax) instead of continuous shares.
    Cached, so that repeated backward passes (GPDP iterations) reuse the compiled functions.
    """
    eps, probs = shock_nodes(n_nodes)
    grid = env.params.discrete_actions

    def _one(aggregate_s, next_aggregate_s, gp_next, terminal_next):
        W = continuation(
            env, aggregate_s, next_aggregate_s, gp_next, eps, probs, terminal_next, gamma
        )
        q_fn = lambda s: gc.q_values(env, s, aggregate_s, W, gamma)  # noqa: E731
        if grid_only:
            return gc.best_grid_share(q_fn, grid, env.n_states)
        return gc.best_share(q_fn, grid, env.n_states, n_iter)

    with_gp = jax.jit(
        lambda a, an, gp: jax.vmap(lambda x, y: _one(x, y, gp, False))(a, an)
    )
    terminal = jax.jit(
        lambda a, an: jax.vmap(lambda x, y: _one(x, y, None, True))(a, an)
    )
    return with_gp, terminal


def _period(tree, t):
    return jax.tree.map(lambda x: x[t], tree)


def backward_pass(
    env,
    aggregate_s_seq,
    gamma,
    n_nodes: int = 5,
    n_iter: int = 30,
    gp_steps_first: int = 300,
    gp_steps: int = 50,
    gp_lr: float = 0.05,
    grid_only: bool = False,
):
    """
    Backward induction over the stations (aggregate_s_seq: leading axes (T, N)).
    Returns:
        gps: list with the V-GP of each period t = 1 .. T-2 (None elsewhere: period T-1 is exact,
             and all stations are identical at period 0)
        shares, values: best shares and exact values at the stations, (T-1, N, n_states), t = 0 .. T-2
        nll: final GP negative log marginal likelihood per period (None where no GP)
    """
    T = aggregate_s_seq.time.shape[0]
    with_gp, terminal = make_lookahead(env, gamma, n_nodes, n_iter, grid_only)
    gps, shares, values, nll = [None] * T, [None] * T, [None] * T, [None] * T
    hyp = init_hyper(2)
    for t in range(T - 2, -1, -1):
        now, nxt = _period(aggregate_s_seq, t), _period(aggregate_s_seq, t + 1)
        if t == T - 2:
            shares[t], values[t] = terminal(now, nxt)
        else:
            shares[t], values[t] = with_gp(now, nxt, gps[t + 1])
        if t >= 1:
            gps[t], nll_t = fit_gp_bounded(
                aggregate_features(env, now),
                values[t],
                hyp,
                gp_steps_first if t == T - 2 else gp_steps,
                gp_lr,
            )
            nll[t] = float(nll_t)
            hyp = gps[t].hyp  # warm start the next (earlier) period
    return gps, jnp.stack(shares[: T - 1]), jnp.stack(values[: T - 1]), nll


# --- a single value GP for all periods: inputs (z, K, T - t), data growing during the backward pass ---
# The training set grows by N points per period. To avoid recompiling at every size, it is padded to
# a few bucket sizes, and padded rows are masked: their kernel rows are zero and their diagonal is 1,
# with zero targets, so they get zero weight and do not change the fit or the predictions.
def _remaining_discount(time_left, gamma):
    """D = sum_{j < time_left} gamma^j: the weight of a constant consumption over the remaining periods."""
    return (1.0 - gamma ** time_left) / (1.0 - gamma)


def value_to_log_ce(V, time_left, gamma, sigma):
    """log c_bar such that V = D u(c_bar), u(c) = c^(1 - sigma) / (1 - sigma)."""
    D = _remaining_discount(time_left, gamma)
    return jnp.log((1.0 - sigma) * V / D) / (1.0 - sigma)


def log_ce_to_value(y, time_left, gamma, sigma):
    D = _remaining_discount(time_left, gamma)
    return D * jnp.exp((1.0 - sigma) * y) / (1.0 - sigma)


# --- tighter bounds for the single GP: with ~1000 points, n s_f^2 / (s_n^2 + jitter) must stay
#     well below float32's limit for a Cholesky factorization ---
SINGLE_LOG_SN_BOUNDS = (float(np.log(3e-2)), 0.0)
SINGLE_LOG_SF_BOUNDS = (float(np.log(0.1)), float(np.log(2.0)))


def _clip_hyper_single(hyp: GPHyper) -> GPHyper:
    return GPHyper(
        log_ell=jnp.clip(hyp.log_ell, *LOG_ELL_BOUNDS),
        log_sf=jnp.clip(hyp.log_sf, *SINGLE_LOG_SF_BOUNDS),
        log_sn=jnp.clip(hyp.log_sn, *SINGLE_LOG_SN_BOUNDS),
    )


def _bucket(n: int) -> int:
    return max(64, 1 << (n - 1).bit_length())


def _masked_standardize(X_raw, Y_raw, mask):
    m = mask[:, None].astype(X_raw.dtype)
    n = m.sum()
    x_mean = (X_raw * m).sum(0) / n
    x_std = jnp.sqrt((((X_raw - x_mean) ** 2) * m).sum(0) / n)
    x_std = jnp.where(x_std > 1e-12, x_std, 1.0)
    y_mean = (Y_raw * m).sum(0) / n
    y_std = jnp.sqrt((((Y_raw - y_mean) ** 2) * m).sum(0) / n)
    y_std = jnp.maximum(y_std, 1e-8 + 1e-6 * jnp.abs(y_mean))
    return (X_raw - x_mean) / x_std * m, (Y_raw - y_mean) / y_std * m, x_mean, x_std, y_mean, y_std


def _masked_gram(X, hyp: GPHyper, mask):
    m = mask.astype(X.dtype)
    K = kernel(X, X, hyp) * m[:, None] * m[None, :]
    return K + jnp.diag((jnp.exp(2 * hyp.log_sn) + JITTER) * m + (1.0 - m))


def _masked_nll(hyp: GPHyper, X, Y, mask):
    L = jnp.linalg.cholesky(_masked_gram(X, hyp, mask))
    alpha = cho_solve((L, True), Y)
    return (0.5 * jnp.sum(Y * alpha) / Y.shape[1] + jnp.sum(jnp.log(jnp.diag(L)))) / mask.sum()


@partial(jax.jit, static_argnames=("n_steps",))
def fit_hyper_masked(X_raw, Y_raw, mask, hyp_init: GPHyper, n_steps: int, lr: float = 0.05):
    """Bounded marginal-likelihood fit of the hyperparameters on the unmasked rows (projected Adam)."""
    X, Y, *_ = _masked_standardize(X_raw, Y_raw, mask)
    hyp_init = _clip_hyper_single(hyp_init)
    tx = optax.adam(lr)
    grad_fn = jax.grad(_masked_nll)

    def _step(carry, _):
        hyp, opt_state = carry
        updates, opt_state = tx.update(grad_fn(hyp, X, Y, mask), opt_state)
        return (_clip_hyper_single(optax.apply_updates(hyp, updates)), opt_state), None

    (hyp, _), _ = jax.lax.scan(_step, (hyp_init, tx.init(hyp_init)), None, length=n_steps)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(hyp)]))
    return jax.tree.map(lambda new, old: jnp.where(finite, new, old), hyp, hyp_init)


@jax.jit
def solve_masked(X_raw, Y_raw, mask, hyp: GPHyper):
    """GP weights for the unmasked rows, with given hyperparameters (masked rows get zero weight)."""
    X, Y, x_mean, x_std, y_mean, y_std = _masked_standardize(X_raw, Y_raw, mask)
    hyp = _clip_hyper_single(hyp)
    L = jnp.linalg.cholesky(_masked_gram(X, hyp, mask))
    return GPState(hyp, X, cho_solve((L, True), Y), x_mean, x_std, y_mean, y_std)


def backward_pass_single(
    env,
    aggregate_s_seq,
    gamma,
    n_nodes: int = 5,
    n_iter: int = 30,
    hyp_init=None,
    hyp_steps_first: int = 300,
    hyp_steps: int = 30,
    fit_at: tuple = (4, 8, 16, 32, 64, 96),
    n_hyp_cols: int = 125,
    gp_lr: float = 0.05,
    grid_only: bool = False,
):
    """
    Backward induction with ONE value GP for all periods, inputs (z, K, T - t), instead of one GP per
    period. After period t is solved at the stations, its values join the GP's training set, so
    period t - 1 uses a GP trained on periods t .. T-2 (8 x 126 = 1008 points at the end).
    - Targets are log consumption equivalents, log c_bar with V = D(T - t) u(c_bar) and D the
      remaining discount sum: raw values grow about 20-fold with the time left, and that trend
      alone pushed the hyperparameters to a kernel matrix that float32 cannot factorize.
    - Hyperparameters are fitted once the training set has `fit_at` periods (not on the first
      period alone, whose points all share the same time left), on a fixed subset of `n_hyp_cols`
      output columns to keep it cheap; in between, only the weights are re-solved.
    - hyp_init: hyperparameters from a previous backward pass (GPDP iteration), for a warm start.
    Returns (gps, shares, values, hyp): gps is a list in which every entry is the final GP (trained on
    all periods), so that greedy_shares works unchanged.
    """
    T = aggregate_s_seq.time.shape[0]
    with_gp, terminal = make_lookahead(env, gamma, n_nodes, n_iter, grid_only)
    cols = jnp.arange(0, env.n_states, max(1, env.n_states // n_hyp_cols))[:n_hyp_cols]
    hyp = init_hyper(3, 3) if hyp_init is None else hyp_init
    X_list, V_list = [], []
    gp, shares, values = None, [None] * T, [None] * T
    for t in range(T - 2, -1, -1):
        now, nxt = _period(aggregate_s_seq, t), _period(aggregate_s_seq, t + 1)
        if t == T - 2:
            shares[t], values[t] = terminal(now, nxt)
        else:
            shares[t], values[t] = with_gp(now, nxt, gp)
        if t >= 1:
            features = aggregate_features(env, now)
            time_left = jnp.full((features.shape[0], 1), T - t, dtype=features.dtype)
            X_list.append(jnp.concatenate([features, time_left], axis=-1))
            V_list.append(value_to_log_ce(values[t], T - t, gamma, env.params.sigma))
            X_raw, Y_raw = jnp.concatenate(X_list), jnp.concatenate(V_list)
            n, size = X_raw.shape[0], _bucket(X_raw.shape[0])
            X_pad = jnp.zeros((size, X_raw.shape[1]), X_raw.dtype).at[:n].set(X_raw)
            Y_pad = jnp.zeros((size, Y_raw.shape[1]), Y_raw.dtype).at[:n].set(Y_raw)
            mask = jnp.arange(size) < n
            n_periods = (T - 2) - t + 1  # periods in the training set, this one included
            if n_periods in fit_at:
                steps = hyp_steps_first if (n_periods == fit_at[0] and hyp_init is None) else hyp_steps
                hyp = fit_hyper_masked(X_pad, Y_pad[:, cols], mask, hyp, steps, gp_lr)
            new_gp = solve_masked(X_pad, Y_pad, mask, hyp)
            if gp is not None and not bool(jnp.all(jnp.isfinite(new_gp.alpha))):
                # --- safeguard: keep the previous GP (trained on later periods) rather than NaN ---
                print(f"single value GP: non-finite solve at period {t}, keeping the previous GP", flush=True)
                hyp = gp.hyp
            else:
                gp = new_gp
    return [gp] * T, jnp.stack(shares[: T - 1]), jnp.stack(values[: T - 1]), hyp


def greedy_shares(env, aggregate_s_seq, gps, gamma, n_nodes: int = 5, n_iter: int = 30, grid_only: bool = False):
    """
    The best-response shares implied by the V-GPs on any paths (not only the stations): at each
    period, a one-period lookahead with the next period's GP. Uses only information available at t.
    Returns (T, E, n_states); the last period's share is irrelevant (terminal reward) and set to 0.5.
    """
    T = aggregate_s_seq.time.shape[0]
    with_gp, terminal = make_lookahead(env, gamma, n_nodes, n_iter, grid_only)
    out = []
    for t in range(T - 1):
        now, nxt = _period(aggregate_s_seq, t), _period(aggregate_s_seq, t + 1)
        s, _ = terminal(now, nxt) if t == T - 2 else with_gp(now, nxt, gps[t + 1])
        out.append(s)
    out.append(jnp.full_like(out[-1], 0.5))
    return jnp.stack(out)
