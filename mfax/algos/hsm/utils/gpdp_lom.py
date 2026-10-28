"""
GPDP with a learned law of motion for the aggregate state (GPDP paper, section 4 and Appendix A).

Law of motion: K_{t+1} - K_t is a GP over (z_t, K_t, T - t), learned from the forward pass's
transitions (a Krusell-Smith style perceived law, learned instead of assumed log-linear). Tomorrow's
shock z' = rho z + nu eps is known in distribution and the time left drops by one, so tomorrow's input
to the value GP is Gaussian:

    x' = (z', K', T - t - 1) ~ N( (rho z, K + m(x), T - t - 1), diag(nu^2, s^2(x), 0) )

with m, s^2 the law-of-motion GP's predictive mean and variance (its noise included).

Closed-form expectation (Appendix A, eq. 16): for a squared-exponential kernel and a Gaussian input,
the expectation of a GP's prediction is exact, E[f(x')] = y_mean + y_std * l^T alpha, with

    l_i = s_f^2 |S L^-1 + I|^(-1/2) exp(-1/2 (x_i - m)^T (S + L)^-1 (x_i - m)),   L = diag(ell^2).

One l (one per training point) serves all 1000 outputs. It replaces the Gauss-Hermite points over z'
and the exact K' of gpdp_backward.py. The outputs need not be Gaussian; only the input must be.

The value GP of the report configuration predicts log c_bar, and V = D u(c_bar) is a curved function
of it: E[V] is not D u(exp(E[log c_bar])). With y = log c_bar approximately Gaussian, E[V] =
D u(exp(E y)) exp((1 - sigma)^2 Var[y] / 2) (lognormal moment); E y is the closed form above and Var y
comes from a first-order expansion of the GP mean around m (cheap; the correction is itself small).
A value GP predicting V directly (one per period) needs no correction.

A household's reward and next individual state depend on the aggregate state only through the prices
(r, w) and the date, and the prices are a function of (z, K) because average income is constant. So
the backward pass needs no mean field: only (z, K, t) at each support point.
"""

from dataclasses import dataclass
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.linalg
from jax.experimental import enable_x64
from jax.scipy.linalg import cho_solve

from mfax.algos.hsm.utils import gpdp_backward as gb
from mfax.algos.hsm.utils import gpdp_continuous as gc
from mfax.algos.hsm.utils.gp_critic import GPHyper, GPState, aggregate_features, init_hyper, kernel


# --- prices at any (z, K) ---
def average_income(env, mu):
    """L = average income under mu (constant along a path: the income chain keeps its marginal)."""
    mu = mu.reshape(*mu.shape[:-1], *env.params.num_states)
    return jnp.sum(mu.sum(-2) * env.params.discrete_states[1], axis=-1)


def aggregate_at(env, template, z, K, time, L):
    """
    An aggregate state with shock z, capital K, date `time` and the prices they imply (the formula
    of env._prices). Its mu is the template's and is NOT consistent with K: a household's reward and
    transition never read mu, only the prices and the date. Works elementwise on batches.
    """
    a = env.params.cobb_douglas_alpha
    interest_rate = jnp.exp(z) * a * (L / (K + 1e-6)) ** (1 - a)
    wage = jnp.exp(z) * (1 - a) * (K / (L + 1e-6)) ** a
    time = jnp.broadcast_to(jnp.asarray(time, template.time.dtype), template.time.shape)
    return template.replace(z=z, time=time, interest_rate=interest_rate, wage=wage)


# --- the law of motion ---
# K falls steeply in the last periods (households run down their wealth before the end), so the GP
# sees log(T - t) instead of T - t: the end of the horizon gets the resolution it needs (R^2 of
# K_{t+1} - K_t on fresh economies 0.997 -> 0.999996). It is fitted and evaluated in float64: its
# targets are almost noiseless, and in float32 the Cholesky factorization fails as the noise shrinks,
# and the predictive variance (s_f^2 minus a nearly equal term) is lost to rounding.
LOM_LOG_SN_BOUNDS = (float(np.log(1e-4)), 0.0)
LOM_LOG_SF_BOUNDS = (float(np.log(0.1)), float(np.log(10.0)))
LOM_LOG_ELL_BOUNDS = (float(np.log(0.05)), float(np.log(100.0)))


@dataclass
class LawOfMotion:
    """GP over (z, K, log(T - t)) -> K_{t+1} - K_t, standardized; numpy float64 arrays."""

    X: np.ndarray  # (n, 3) standardized inputs
    alpha: np.ndarray  # (n,)
    chol: np.ndarray  # (n, n) Cholesky factor of the Gram matrix (noise included)
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    hyp: GPHyper  # float64


def _lom_inputs(X_raw):
    X_raw = np.asarray(X_raw, np.float64)
    return np.column_stack([X_raw[:, 0], X_raw[:, 1], np.log(X_raw[:, 2])])


def _clip_lom(hyp: GPHyper) -> GPHyper:
    return GPHyper(
        log_ell=jnp.clip(hyp.log_ell, *LOM_LOG_ELL_BOUNDS),
        log_sf=jnp.clip(hyp.log_sf, *LOM_LOG_SF_BOUNDS),
        log_sn=jnp.clip(hyp.log_sn, *LOM_LOG_SN_BOUNDS),
    )


def _lom_gram(X, hyp):
    return kernel(X, X, hyp) + (jnp.exp(2 * hyp.log_sn) + 1e-10) * jnp.eye(X.shape[0])


def _lom_nll(hyp, X, y):
    L = jnp.linalg.cholesky(_lom_gram(X, hyp))
    return (0.5 * y @ cho_solve((L, True), y) + jnp.sum(jnp.log(jnp.diag(L)))) / X.shape[0]


@partial(jax.jit, static_argnames=("n_steps",))
def _fit_lom_hyper(X, y, hyp, n_steps, lr):
    """Projected Adam. A step from hyperparameters with a non-finite NLML or gradient is not taken, and
    the result is the last hyperparameters whose NLML was finite (the final update is not checked by
    the loop: once it gave a matrix that the float64 Cholesky could not factorize)."""
    tx = optax.adam(lr)
    grad_fn = jax.value_and_grad(_lom_nll)

    def _step(carry, _):
        hyp, opt_state, good = carry
        nll, grads = grad_fn(hyp, X, y)
        updates, new_state = tx.update(grads, opt_state)
        new = _clip_lom(optax.apply_updates(hyp, updates))
        ok = jnp.isfinite(nll) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads)]))
        pick = lambda a, b: jax.tree.map(lambda u, v: jnp.where(ok, u, v), a, b)  # noqa: E731
        return (pick(new, hyp), pick(new_state, opt_state), pick(hyp, good)), None

    hyp = _clip_lom(hyp)
    (hyp, _, good), _ = jax.lax.scan(_step, (hyp, tx.init(hyp), hyp), None, length=n_steps)
    nll = _lom_nll(hyp, X, y)
    hyp = jax.tree.map(lambda u, v: jnp.where(jnp.isfinite(nll), u, v), hyp, good)
    return hyp, _lom_nll(hyp, X, y)


def transitions(env, aggregate_s_seq):
    """
    Law-of-motion training pairs from simulated paths (leading axes (T, N)), t = 0 .. T-2:
    X = (z_t, K_t, T - t) ((T - 1) N, 3), y = K_{t+1} - K_t ((T - 1) N, 1).
    """
    T = env.params.max_steps_in_episode
    f = jax.vmap(lambda a: aggregate_features(env, a))(aggregate_s_seq)  # (T, N, 2)
    time_left = (T - aggregate_s_seq.time).astype(f.dtype)[..., None]  # (T, N, 1)
    X = jnp.concatenate([f[:-1], time_left[:-1]], axis=-1).reshape(-1, 3)
    y = (f[1:, :, 1] - f[:-1, :, 1]).reshape(-1, 1)
    return X, y


def fit_law_of_motion(X, y, hyp=None, n_steps: int = 300, lr: float = 0.05):
    """
    Bounded marginal-likelihood fit (projected Adam, a step that gives a non-finite value is skipped),
    in float64. X: (n, 3) raw inputs (z, K, T - t); y: (n, 1) K_{t+1} - K_t. hyp: warm start.
    Returns (LawOfMotion, final NLML).
    """
    A = _lom_inputs(X)
    b = np.asarray(y, np.float64)[:, 0]
    x_mean, x_std = A.mean(0), np.where(A.std(0) > 1e-12, A.std(0), 1.0)
    y_mean, y_std = float(b.mean()), float(max(b.std(), 1e-12))
    A, b = (A - x_mean) / x_std, (b - y_mean) / y_std
    with enable_x64():
        if hyp is None:
            hyp = GPHyper(jnp.zeros(3), jnp.asarray(0.0), jnp.asarray(np.log(0.1)))
        hyp = jax.tree.map(lambda h: jnp.asarray(h, jnp.float64), hyp)
        hyp, nll = _fit_lom_hyper(jnp.asarray(A), jnp.asarray(b), hyp, n_steps, lr)
        for _ in range(4):  # last resort: more noise until the Gram matrix factorizes
            L = jnp.linalg.cholesky(_lom_gram(jnp.asarray(A), hyp))
            alpha = cho_solve((L, True), jnp.asarray(b))
            if bool(jnp.all(jnp.isfinite(alpha))):
                break
            print("law of motion: Cholesky failed, noise x10", flush=True)
            hyp = hyp.replace(log_sn=hyp.log_sn + np.log(10.0))
            nll = _lom_nll(hyp, jnp.asarray(A), jnp.asarray(b))
        lom = LawOfMotion(A, np.asarray(alpha), np.asarray(L), x_mean, x_std, y_mean, y_std,
                          jax.tree.map(np.asarray, hyp))
    return lom, float(nll)


def predict_next_K(lom: LawOfMotion, X_raw):
    """(m, 3) inputs (z, K, T - t) -> mean and variance of K_{t+1}, each (m,) float64. The variance
    includes the GP's noise: the part of K_{t+1} that (z, K, T - t) does not determine (the rest of mu)."""
    A = (_lom_inputs(X_raw) - lom.x_mean) / lom.x_std
    ell2, sf2, sn2 = (np.exp(2 * np.asarray(h, np.float64)) for h in (lom.hyp.log_ell, lom.hyp.log_sf, lom.hyp.log_sn))
    k = sf2 * np.exp(-0.5 * (((A[:, None, :] - lom.X[None]) ** 2) / ell2).sum(-1))  # (m, n)
    mean = lom.y_mean + lom.y_std * (k @ lom.alpha)
    v = scipy.linalg.solve_triangular(lom.chol, k.T, lower=True)  # (n, m)
    var = sf2 - np.sum(v**2, axis=0) + sn2
    return np.asarray(X_raw, np.float64)[:, 1] + mean, lom.y_std**2 * np.maximum(var, 0.0)


# --- closed-form expectation of a GP's prediction under a Gaussian input (Appendix A) ---
def expected_prediction(gp: GPState, mean_raw, var_raw):
    """
    For one Gaussian input x ~ N(mean_raw, diag(var_raw)) ((D,) each): E[f(x)] of the GP's mean
    prediction f, exact (eq. 16), and Var[f(x)] to first order. Both (J,), one per output.
    Padded training rows (zero weights, gpdp_backward.solve_masked) contribute nothing.
    """
    m = (mean_raw - gp.x_mean) / gp.x_std
    S = var_raw / gp.x_std**2
    ell2 = jnp.exp(2 * gp.hyp.log_ell)
    sf2 = jnp.exp(2 * gp.hyp.log_sf)
    d = gp.X - m  # (n, D)
    l = sf2 * jnp.prod(1.0 + S / ell2) ** -0.5 * jnp.exp(-0.5 * jnp.sum(d**2 / (S + ell2), -1))
    mean = gp.y_mean + gp.y_std * (l @ gp.alpha)
    k = sf2 * jnp.exp(-0.5 * jnp.sum(d**2 / ell2, -1))  # k(m, x_i)
    grad = (k[:, None] * d / ell2).T @ gp.alpha  # (D, J): gradient of the standardized mean at m
    var = gp.y_std**2 * jnp.sum(S[:, None] * grad**2, axis=0)
    return mean, var


def continuation(env, aggregate_s, K_mean, K_var, gp_next, gamma):
    """
    W(s') = E[V_{t+1}(s'; z', K', T - t - 1)] for every individual state s' of one economy, with
    z' ~ N(rho z, nu^2) and K' ~ N(K_mean, K_var) independent (K' is fixed before z' is drawn).
    gp_next: the single value GP (inputs (z, K, T - t), predicts log c_bar) or a per-period value GP
    (inputs (z, K), predicts V: the closed form is then E[V] itself).
    """
    p = env.params
    z_mean, z_var = p.rho * aggregate_s.z, p.nu**2
    if gp_next.X.shape[-1] == 3:
        time_left = p.max_steps_in_episode - (aggregate_s.time + 1)
        mean_in = jnp.stack([z_mean, K_mean, jnp.asarray(time_left, K_mean.dtype)])
        var_in = jnp.stack([z_var, K_var, 0.0])
        y_mean, y_var = expected_prediction(gp_next, mean_in, var_in)
        V = gb.log_ce_to_value(y_mean, time_left, gamma, p.sigma)
        return V * jnp.exp(0.5 * (1.0 - p.sigma) ** 2 * y_var)  # lognormal moment of D u(exp(y))
    V, _ = expected_prediction(gp_next, jnp.stack([z_mean, K_mean]), jnp.stack([z_var, K_var]))
    return V


def terminal_continuation(env, aggregate_s, K_mean, K_var, L, eps, probs):
    """
    W(s') = E[terminal reward of s' at prices (z', K')] (the next period is the last one). The terminal
    reward is a known function, not a GP, so the expectation uses Gauss-Hermite points in z' and K'.
    """
    p = env.params
    z_next = p.rho * aggregate_s.z + p.nu * eps
    K_next = K_mean + jnp.sqrt(K_var) * eps
    zz, KK = (x.ravel() for x in jnp.meshgrid(z_next, K_next, indexing="ij"))
    ww = jnp.outer(probs, probs).ravel()
    half = jnp.full((env.n_states,), 0.5)  # the share does not matter for the terminal reward
    values = jax.vmap(
        lambda z, K: gc.reward(env, half, aggregate_at(env, aggregate_s, z, K, aggregate_s.time + 1, L), True)
    )(zz, KK)
    return ww @ values


@lru_cache(maxsize=None)
def make_lookahead(env, gamma, n_nodes: int = 5, n_iter: int = 30):
    """
    Jitted one-period best response for a batch of support points (leading axis E):
    (aggregate_s, K_mean (E,), K_var (E,), L (E,)[, gp_next]) -> (shares (E, n_states), values (E, n_states)).
    As gpdp_backward.make_lookahead, with tomorrow's K from the law of motion.
    """
    eps, probs = gb.shock_nodes(n_nodes)
    grid = env.params.discrete_actions

    def _one(aggregate_s, K_mean, K_var, L, gp_next, terminal_next):
        if terminal_next:
            W = terminal_continuation(env, aggregate_s, K_mean, K_var, L, eps, probs)
        else:
            W = continuation(env, aggregate_s, K_mean, K_var, gp_next, gamma)
        q_fn = lambda s: gc.q_values(env, s, aggregate_s, W, gamma)  # noqa: E731
        return gc.best_share(q_fn, grid, env.n_states, n_iter)

    with_gp = jax.jit(
        lambda a, m, v, L, gp: jax.vmap(lambda *x: _one(*x, gp, False))(a, m, v, L)
    )
    terminal = jax.jit(lambda a, m, v, L: jax.vmap(lambda *x: _one(*x, None, True))(a, m, v, L))
    return with_gp, terminal


# --- support points ---
def support_from_paths(env, aggregate_s_seq):
    """(z, K) of simulated paths (the stations): Z, K each (T, N); L (N,) from the first period."""
    f = jax.vmap(lambda a: aggregate_features(env, a))(aggregate_s_seq)
    return f[..., 0], f[..., 1], average_income(env, aggregate_s_seq.mu[0])


def support_from_law_of_motion(env, lom: LawOfMotion, z0, K0, seed, n_paths):
    """
    Support paths simulated with the learned law of motion instead of the push-forward of mu:
    z follows its AR(1), K_{t+1} = K_t + m + s eps_K, from (z0, K0) (the reset). Returns Z, K, each
    (T, n_paths) float32.
    """
    T, p = env.params.max_steps_in_episode, env.params
    rng = np.random.default_rng(seed)
    Z, K = np.zeros((T, n_paths)), np.zeros((T, n_paths))
    Z[0], K[0] = z0, K0
    for t in range(T - 1):
        K_mean, K_var = predict_next_K(lom, np.column_stack([Z[t], K[t], np.full(n_paths, T - t)]))
        Z[t + 1] = p.rho * Z[t] + p.nu * rng.standard_normal(n_paths)
        K[t + 1] = K_mean + np.sqrt(K_var) * rng.standard_normal(n_paths)
    return jnp.asarray(Z, jnp.float32), jnp.asarray(K, jnp.float32)


def policy_training_data(Z, K, shares, T):
    """Policy GP pairs at the support points: inputs (z, K, T - t), logit targets; t = 0 .. T-2."""
    from mfax.algos.hsm.utils.gpdp_policy import logit

    n_periods = shares.shape[0]
    time_left = jnp.broadcast_to((T - jnp.arange(n_periods))[:, None], Z[:n_periods].shape)
    X = jnp.stack([Z[:n_periods], K[:n_periods], time_left.astype(Z.dtype)], axis=-1)
    return X.reshape(-1, 3), logit(shares).reshape(-1, shares.shape[-1])


# --- backward pass ---
def backward_pass_single(
    env,
    template,
    Z,
    K,
    L,
    lom: LawOfMotion,
    gamma,
    n_nodes: int = 5,
    n_iter: int = 30,
    hyp_init=None,
    hyp_steps_first: int = 300,
    hyp_steps: int = 30,
    fit_at: tuple = (4, 8, 16, 32, 64, 96),
    n_hyp_cols: int = 125,
    gp_lr: float = 0.05,
):
    """
    gpdp_backward.backward_pass_single (one value GP over (z, K, T - t), log c_bar targets, same fit
    schedule) with tomorrow's K from the law of motion and the closed-form expectation over (z', K').
    The Bellman equation is solved at support points given only by Z, K ((T, N) each) and the date.
    template: any aggregate state with leading axis N (its mu is never read); L: average income (N,).
    Returns (gps, shares (T-1, N, n_states), values (T-1, N, n_states), hyp), as backward_pass_single.
    """
    T = Z.shape[0]
    N = Z.shape[1]
    with_gp, terminal = make_lookahead(env, gamma, n_nodes, n_iter)
    cols = jnp.arange(0, env.n_states, max(1, env.n_states // n_hyp_cols))[:n_hyp_cols]
    hyp = init_hyper(3, 3) if hyp_init is None else hyp_init
    X_list, V_list = [], []
    gp, shares, values = None, [None] * T, [None] * T
    for t in range(T - 2, -1, -1):
        now = aggregate_at(env, template, Z[t], K[t], t, L)
        X_t = jnp.stack([Z[t], K[t], jnp.full((N,), T - t, Z.dtype)], axis=-1)
        K_mean, K_var = (jnp.asarray(a, jnp.float32) for a in predict_next_K(lom, X_t))
        if t == T - 2:
            shares[t], values[t] = terminal(now, K_mean, K_var, L)
        else:
            shares[t], values[t] = with_gp(now, K_mean, K_var, L, gp)
        if t >= 1:
            X_list.append(X_t)
            V_list.append(gb.value_to_log_ce(values[t], T - t, gamma, env.params.sigma))
            X_raw, Y_raw = jnp.concatenate(X_list), jnp.concatenate(V_list)
            n, size = X_raw.shape[0], gb._bucket(X_raw.shape[0])
            X_pad = jnp.zeros((size, X_raw.shape[1]), X_raw.dtype).at[:n].set(X_raw)
            Y_pad = jnp.zeros((size, Y_raw.shape[1]), Y_raw.dtype).at[:n].set(Y_raw)
            mask = jnp.arange(size) < n
            n_periods = (T - 2) - t + 1
            if n_periods in fit_at:
                steps = hyp_steps_first if (n_periods == fit_at[0] and hyp_init is None) else hyp_steps
                hyp = gb.fit_hyper_masked(X_pad, Y_pad[:, cols], mask, hyp, steps, gp_lr)
            new_gp = gb.solve_masked(X_pad, Y_pad, mask, hyp)
            if gp is not None and not bool(jnp.all(jnp.isfinite(new_gp.alpha))):
                print(f"single value GP: non-finite solve at period {t}, keeping the previous GP", flush=True)
                hyp = gp.hyp
            else:
                gp = new_gp
    return [gp] * T, jnp.stack(shares[: T - 1]), jnp.stack(values[: T - 1]), hyp
