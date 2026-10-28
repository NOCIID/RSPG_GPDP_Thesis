"""
Continuous consumption shares for the push-forward endogenous environment: one share per individual
state, instead of a probability over the discrete actions.

The environment's generic loops (mf_transition, mf_expected_value, mf_reward) evaluate every
discrete action. The functions below call the environment's per-state functions instead
(_single_pushforward_step, _single_pushforward_reward, _prices), which already accept a continuous
action, so no environment code is changed. With shares equal to grid actions they reproduce the
discrete computations exactly (checked in experiments/gpdp/step1_continuous_br.py).

Used by GPDP (see FINDINGS.md, "Design: GPDP instead of gradient descent").
"""

import jax
import jax.numpy as jnp

GOLDEN = 0.6180339887498949  # (sqrt(5) - 1) / 2


# --- mean-field dynamics with one share per state ---
def _per_state(env, shares, aggregate_s):
    """Next-state indices and probabilities of every individual state under its own share."""
    return jax.vmap(env._single_pushforward_step, in_axes=(0, 0, None))(
        env.state_indices, shares[:, None], aggregate_s
    )  # each (n_states, n_next)


def transition(env, mu, shares, aggregate_s):
    """Push-forward of the mean field, every state using its own share. Returns next mu (n_states,)."""
    idxs, probs = _per_state(env, shares, aggregate_s)
    return (
        jnp.zeros((env.n_states,))
        .at[idxs.reshape(-1)]
        .add((mu[:, None] * probs).reshape(-1))
    )


def expected_value(env, vec, shares, aggregate_s):
    """E[vec(x') | x, share(x)] for every individual state x. Returns (n_states,)."""
    idxs, probs = _per_state(env, shares, aggregate_s)
    return jnp.sum(vec[idxs] * probs, axis=-1)


def reward(env, shares, aggregate_s, terminal):
    """Reward of every individual state under its own share; the terminal reward on the last step.
    (The endogenous reward does not depend on the next aggregate state, so the current one is passed.)"""
    r_step, r_term = jax.vmap(
        env._single_pushforward_reward, in_axes=(0, 0, None, None)
    )(env.state_indices, shares[:, None], aggregate_s, aggregate_s)
    return jnp.where(terminal, r_term, r_step)


def step(env, key, aggregate_s, shares):
    """
    One environment step with one share per state: the continuous counterpart of
    PushforwardEndogenousEnvironment.mf_step_env (no auto-reset). `key` plays the role of the
    `key_step` that mf_step passes to mf_step_env.
    Returns (next_aggregate_s, rewards (n_states,), terminated).
    """
    p = env.params
    next_mu = transition(env, aggregate_s.mu, shares, aggregate_s)
    next_z = p.rho * aggregate_s.z + p.nu * jax.random.normal(key)
    next_interest_rate, next_wage = env._prices(next_mu, next_z)
    next_time = aggregate_s.time + 1
    next_aggregate_s = aggregate_s.replace(
        mu=next_mu,
        z=next_z,
        time=next_time,
        interest_rate=next_interest_rate,
        wage=next_wage,
    )
    terminated = env.is_terminal(next_time)
    return next_aggregate_s, reward(env, shares, aggregate_s, terminated), terminated


# --- choosing shares ---
def q_values(env, shares, aggregate_s, v_next, gamma):
    """Q of every individual state for its own share: reward + gamma * E[v_next(x')].
    On the last step the reward is the terminal one and there is no continuation."""
    terminal = env.is_terminal(aggregate_s.time + 1)
    continuation = jnp.where(
        terminal, 0.0, expected_value(env, v_next, shares, aggregate_s)
    )
    return reward(env, shares, aggregate_s, terminal) + gamma * continuation


def golden_max(f, lo, hi, n_iter):
    """
    Elementwise golden-section search for the maximum of f on [lo, hi]: f maps shares (n,) to values
    (n,), and each element is searched on its own interval. One evaluation of f per iteration; the
    interval shrinks by 0.618 each time. Finds the maximum of a unimodal f (Q is concave in the share
    here). Returns (argmax (n,), max (n,)).
    """
    c = hi - GOLDEN * (hi - lo)
    d = lo + GOLDEN * (hi - lo)
    carry = (lo, hi, c, d, f(c), f(d))

    def _iteration(_, carry):
        a, b, c, d, fc, fd = carry
        left = fc >= fd  # the maximum lies in [a, d]
        a = jnp.where(left, a, c)
        b = jnp.where(left, d, b)
        x = jnp.where(left, b - GOLDEN * (b - a), a + GOLDEN * (b - a))
        fx = f(x)
        return (
            a,
            b,
            jnp.where(left, x, d),
            jnp.where(left, c, x),
            jnp.where(left, fx, fd),
            jnp.where(left, fc, fx),
        )

    _, _, c, d, fc, fd = jax.lax.fori_loop(0, n_iter, _iteration, carry)
    return jnp.where(fc >= fd, c, d), jnp.maximum(fc, fd)


def best_grid_share(q_fn, grid, n_states):
    """Best share per state among the grid actions only (what the discrete best response does)."""
    q_grid = jax.vmap(lambda g: q_fn(jnp.full((n_states,), g)))(grid)  # (n_grid, n_states)
    return grid[jnp.argmax(q_grid, axis=0)], jnp.max(q_grid, axis=0)


def best_share(q_fn, grid, n_states, n_iter=30, lo=1e-4, hi=1.0):
    """
    Best continuous share per state: golden-section search on [lo, hi], compared with every grid
    action so that the result is never worse than the best grid action. q_fn maps shares (n_states,)
    to Q (n_states,). Returns (shares (n_states,), Q (n_states,)).
    """
    x, q = golden_max(
        q_fn, jnp.full((n_states,), lo), jnp.full((n_states,), hi), n_iter
    )
    g, q_g = best_grid_share(q_fn, grid, n_states)
    use_grid = q_g > q
    return jnp.where(use_grid, g, x), jnp.where(use_grid, q_g, q)


# --- best response along given paths ---
def path_best_response(env, aggregate_s_seq, gamma, grid_only=False, n_iter=30):
    """
    Best response along given mean-field paths by backward induction: the continuous-share
    counterpart of mfax.algos.hsm.best_response.br. Like br, it uses the realized paths, so it knows
    future aggregate shocks (a clairvoyant best response).
    aggregate_s_seq: aggregate states with leading axes (T, E). grid_only=True restricts the choice
    to the grid actions, which reproduces br.
    Returns (V_0 (E, n_states), best shares (T, E, n_states)).
    """
    E = aggregate_s_seq.time.shape[1]
    grid = env.params.discrete_actions

    def _one_period(v_next, aggregate_s):  # aggregate_s: leading axis E
        def _per_env(v_next_e, aggregate_s_e):
            q_fn = lambda s: q_values(env, s, aggregate_s_e, v_next_e, gamma)  # noqa: E731
            if grid_only:
                return best_grid_share(q_fn, grid, env.n_states)
            return best_share(q_fn, grid, env.n_states, n_iter)

        shares, v = jax.vmap(_per_env)(v_next, aggregate_s)
        return v, shares

    v0, shares = jax.lax.scan(
        _one_period, jnp.zeros((E, env.n_states)), aggregate_s_seq, reverse=True
    )
    return v0, shares


def path_evaluate(env, aggregate_s_seq, shares_seq, gamma):
    """
    Value of given shares along given paths, by backward recursion on the realized paths.
    aggregate_s_seq: leading axes (T, E); shares_seq: (T, E, n_states).
    Returns V_t for every period: (T, E, n_states).
    """
    E = aggregate_s_seq.time.shape[1]

    def _one_period(v_next, inputs):
        aggregate_s, shares = inputs
        v = jax.vmap(lambda a, s, vn: q_values(env, s, a, vn, gamma))(
            aggregate_s, shares, v_next
        )
        return v, v

    _, V = jax.lax.scan(
        _one_period,
        jnp.zeros((E, env.n_states)),
        (aggregate_s_seq, shares_seq),
        reverse=True,
    )
    return V
