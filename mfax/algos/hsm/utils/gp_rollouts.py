"""
Rollout utilities for the GP critic.

- MFPolicyFn: one interface for the SPG (feed-forward) and RSPG (recurrent) mean-field policies.
- inf_step: an environment step that never terminates (the infinite-horizon reading of the model).
- value_rollout: simulate long paths and evaluate the policy by backward recursion, recomputing the
  actions in the backward pass instead of storing them (memory-light), as in the SRL notebook.
- replay_hidden: recompute a GRU state from a stored observation history with the current parameters.
"""

from functools import partial

import jax
import jax.numpy as jnp

from mfax.algos.hsm.utils.gp_critic import (
    aggregate_features,
    gp_inputs,
    init_feature_history,
    push_feature_history,
)


def select_rows(mask, a, b):
    """Row-wise select over the leading (economy) axis of two pytrees."""
    return jax.tree.map(
        lambda x, y: jnp.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b
    )


class MFPolicyFn:
    """
    prob_a, next_hidden = policy_fn(params, hidden, aggregate_obs, done)

    For a feed-forward policy, `hidden` is a dummy (E, 1) array passed through unchanged.
    """

    def __init__(
        self,
        env,
        mf_policy_net,
        individual_states,
        normalize_obs: bool,
        normalize_states: bool,
        recurrent: bool,
    ):
        self.env = env
        self.net = mf_policy_net
        self.individual_states = individual_states
        self.normalize_obs = normalize_obs
        self.normalize_states = normalize_states
        self.recurrent = recurrent
        self.hidden_size = (
            mf_policy_net.aggregate_encoder.hidden_size if recurrent else 1
        )

    @classmethod
    def from_wrapper(cls, env, wrapper, state_type: str):
        """Build from a saved MFActorWrapper / MFRecurrentActorWrapper."""
        recurrent = hasattr(wrapper, "init_hidden")
        individual_states = (
            env.state_indices if state_type == "indices" else env.params.states
        )
        return cls(
            env,
            wrapper.mf_policy,
            individual_states,
            wrapper.normalize_obs,
            wrapper.normalize_individual_s,
            recurrent,
        )

    def init_hidden(self, n: int) -> jax.Array:
        return jnp.zeros((n, self.hidden_size), dtype=jnp.float32)

    def _states(self):
        return self.env.normalize_individual_s(
            self.individual_states, self.normalize_states
        )

    def __call__(self, params, hidden, aggregate_obs, done):
        obs = self.env.normalize_obs(aggregate_obs, self.normalize_obs)
        if self.recurrent:
            return self.net.dist_prob(params, self._states(), obs, hidden, done)
        return self.net.dist_prob(params, self._states(), obs), hidden

    def encode(self, params, hidden, aggregate_obs, done):
        """GRU update only (no policy head): next hidden state for a batch of economies."""
        obs = self.env.normalize_obs(aggregate_obs, self.normalize_obs)

        def _single(h, o, d):
            new_h, _ = self.net.aggregate_encoder.apply(
                {"params": params["encoder"]}, h, o, d
            )
            return new_h

        return jax.vmap(_single)(hidden, obs, done)

    def mean_action(self, prob_a):
        """Expected consumption share E[a | x] for each individual state."""
        return prob_a @ self.net.action_grid[:, 0]


def replay_hidden(policy_fn: MFPolicyFn, params, obs_hist, n_valid):
    """
    Recompute the GRU state h_{t-1} from the observations o_0, ..., o_{t-1} since the episode start.

    obs_hist: (E, L, obs_dim), right-aligned (the last n_valid[e] entries are valid).
    n_valid: (E,) number of valid observations per economy. Economies with n_valid = 0 get h = 0.
    Gradients flow through the replay (observations are constants), so the GRU weights receive the
    same gradient as with full backpropagation through time.
    """
    E, L, _ = obs_hist.shape
    h0 = policy_fn.init_hidden(E)
    if L == 0:
        return h0
    done = jnp.zeros((E,), dtype=jnp.bool_)

    def _step(h, inp):
        t, o = inp
        h_new = policy_fn.encode(params, h, o, done)
        valid = t >= (L - n_valid)
        return jnp.where(valid[:, None], h_new, h), None

    h, _ = jax.lax.scan(
        _step, h0, (jnp.arange(L), jnp.swapaxes(obs_hist, 0, 1))
    )
    return h


def inf_step(env, key, aggregate_s, prob_a):
    """
    One push-forward step that never terminates: the time counter is hidden from the environment,
    so there is no terminal reward and no automatic reset.
    """
    obs, next_s, mat_r, _, _ = env.mf_step_env(
        key, aggregate_s.replace(time=jnp.zeros_like(aggregate_s.time)), prob_a
    )
    return obs, next_s.replace(time=aggregate_s.time + 1), mat_r


@partial(
    jax.jit,
    static_argnames=("env", "policy_fn", "length", "return_all", "finite", "time_input"),
)
def value_rollout(
    env,
    policy_fn: MFPolicyFn,
    params,
    gamma: float,
    aggregate_s,
    aggregate_obs,
    hidden,
    feat_hist,
    key,
    length: int,
    return_all: bool,
    tail=None,
    finite: bool = False,
    time_input: bool = False,
):
    """
    Simulate `length` periods for a batch of economies under the policy, then evaluate the policy
    by backward recursion: v_t = E_pi[r_t] + gamma * E[v_{t+1}], for all individual states at once
    (idiosyncratic risk integrated exactly; aggregate path simulated).

    finite=False: the infinite-horizon reading (no terminal period).
    finite=True: the episode ends at T = max_steps_in_episode, as in the environment: the step into
        period T pays the terminal reward and nothing is valued after it (economies may start at
        different periods; periods past T are masked out).
    time_input: GP inputs include the time left in the episode, T - t.

    Returns
        V: (length, E, n_states) if return_all else (E, n_states) at t = 0
        X: (length, E, D) GP inputs along the path (if return_all) else (E, D) at t = 0
        final: (aggregate_s, aggregate_obs, hidden, feat_hist) after `length` periods
    """
    E = aggregate_obs.shape[0]
    T = env.params.max_steps_in_episode
    done = jnp.zeros((E,), dtype=jnp.bool_)
    step = jax.vmap(partial(inf_step, env), in_axes=(0, 0, 0))

    def _fwd(carry, k):
        s, obs, h, fh = carry
        prob_a, h_next = policy_fn(params, h, obs, done)
        next_obs, next_s, _ = step(jax.random.split(k, E), s, prob_a)
        fh_next = push_feature_history(fh, aggregate_features(env, next_s))
        x = gp_inputs(fh, T - s.time if time_input else None)
        return (next_s, next_obs, h_next, fh_next), (s, obs, h, x)

    final, (S, O, H, X) = jax.lax.scan(
        _fwd,
        (aggregate_s, aggregate_obs, hidden, feat_hist),
        jax.random.split(key, length),
    )
    S_next = jax.tree.map(
        lambda a, last: jnp.concatenate([a[1:], last[None]], axis=0), S, final[0]
    )

    def _bwd(v, inp):
        s, s_next, obs, h = inp
        prob_a, _ = policy_fn(params, h, obs, done)
        mat_r, mat_r_term = jax.vmap(env.mf_reward)(s, s_next)
        cont = jax.vmap(env.mf_expected_value)(v, prob_a, s)
        if finite:
            last = (s_next.time >= T)[:, None]
            mat_r = jnp.where(last[..., None], mat_r_term, mat_r)
            cont = jnp.where(last, 0.0, cont)
        v = jnp.sum(prob_a * mat_r, axis=-1) + gamma * cont
        if finite:
            v = jnp.where((s.time >= T)[:, None], 0.0, v)
        return v, (v if return_all else None)

    v_init = jnp.zeros((E, env.n_states)) if tail is None else tail
    v0, V = jax.lax.scan(_bwd, v_init, (S, S_next, O, H), reverse=True)
    if return_all:
        return V, X, final
    return v0, X[0], final


@partial(jax.jit, static_argnames=("env", "policy_fn", "length"))
def forward_rollout(
    env,
    policy_fn: MFPolicyFn,
    params,
    aggregate_s,
    aggregate_obs,
    hidden,
    feat_hist,
    key,
    length: int,
):
    """
    Simulate `length` infinite-horizon periods without evaluating values.
    Returns the final carry and the observations seen (length, E, obs_dim).
    """
    E = aggregate_obs.shape[0]
    done = jnp.zeros((E,), dtype=jnp.bool_)
    step = jax.vmap(partial(inf_step, env), in_axes=(0, 0, 0))

    def _fwd(carry, k):
        s, obs, h, fh = carry
        prob_a, h_next = policy_fn(params, h, obs, done)
        next_obs, next_s, _ = step(jax.random.split(k, E), s, prob_a)
        fh_next = push_feature_history(fh, aggregate_features(env, next_s))
        return (next_s, next_obs, h_next, fh_next), obs

    final, O = jax.lax.scan(
        _fwd,
        (aggregate_s, aggregate_obs, hidden, feat_hist),
        jax.random.split(key, length),
    )
    return final, O


def reset_batch(env, key, n: int, n_lags: int, policy_fn: MFPolicyFn):
    """Reset n economies: (aggregate_s, aggregate_obs, hidden, feat_hist)."""
    obs, s = jax.vmap(env.mf_reset)(jax.random.split(key, n))
    fh = init_feature_history(aggregate_features(env, s), n_lags)
    return s, obs, policy_fn.init_hidden(n), fh
