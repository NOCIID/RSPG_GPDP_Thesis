"""
Policy GP for GPDP (build step 3 in FINDINGS.md).

- Inputs: (z, K, T - t) of an economy at a period.
- Outputs: the consumption share of every individual state (n_states outputs), one shared kernel,
  as in gp_critic.py.
- Targets: the best shares computed by the backward pass at the stations, fitted on
  log(a / (1 - a)), so that every prediction maps back to a share in (0, 1).
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from mfax.algos.hsm.utils.gp_critic import GPState, aggregate_features, gp_predict, init_hyper
from mfax.utils.nets.base import RecurrentEncoder
from mfax.algos.hsm.utils.gpdp_backward import fit_gp_bounded
from mfax.envs.pushforward.utils import distribute

SHARE_CLIP = (1e-4, 0.999)


def logit(a):
    a = jnp.clip(a, *SHARE_CLIP)
    return jnp.log(a) - jnp.log1p(-a)


def policy_inputs(env, aggregate_s, use_time: bool = True):
    """(z, K, T - t), or (z, K) without the date, for a batch of aggregate states (leading axis E)."""
    T = env.params.max_steps_in_episode
    features = aggregate_features(env, aggregate_s)
    if not use_time:
        return features
    time_left = jnp.asarray(T - aggregate_s.time, features.dtype)
    return jnp.concatenate([features, time_left[:, None]], axis=-1)


def uses_time(gp) -> bool:
    """Whether a policy GP takes the time left as an input: (z, K, T - t) has 3 inputs, (z, K) 2."""
    return gp.X.shape[-1] == 3


def training_data(env, aggregate_s_seq, shares, use_time: bool = True):
    """
    Policy GP training pairs from the backward pass: inputs (z, K, T - t) and logit targets.
    aggregate_s_seq: the stations, leading axes (T, N); shares: (T - 1, N, n_states) for t = 0 .. T-2
    (the last period's share does not matter: its reward is the terminal one).
    use_time=False drops the time left from the inputs (the time-input ablation).
    Returns X ((T - 1) N, 3 or 2), Y ((T - 1) N, n_states).
    """
    n_periods = shares.shape[0]
    X = jax.vmap(lambda a: policy_inputs(env, a, use_time))(
        jax.tree.map(lambda x: x[:n_periods], aggregate_s_seq)
    )
    return X.reshape(-1, X.shape[-1]), logit(shares).reshape(-1, shares.shape[-1])


def fit_policy_gp(X, Y, n_steps: int = 300, lr: float = 0.05):
    """Fit the policy GP (bounded hyperparameters, as the V-GPs). Returns (GPState, NLML)."""
    return fit_gp_bounded(X, Y, init_hyper(X.shape[-1], X.shape[-1]), n_steps, lr)


def predict_shares(env, gp, aggregate_s):
    """Shares of every individual state for a batch of economies (leading axis E): (E, n_states)."""
    return jax.nn.sigmoid(gp_predict(gp, policy_inputs(env, aggregate_s, uses_time(gp))))


def path_shares(env, gp, aggregate_s_seq):
    """Shares along whole paths (leading axes (T, E)): (T, E, n_states)."""
    T, E = aggregate_s_seq.time.shape
    flat = jax.tree.map(lambda x: x.reshape(T * E, *x.shape[2:]), aggregate_s_seq)
    return predict_shares(env, gp, flat).reshape(T, E, -1)


# --- MFAX agent interface, for MFAX's own evaluation (exploitability, sequences) ---
def shares_to_grid_probs(shares, grid):
    """Split each share between its two neighbouring grid actions (linear weights, as `distribute`
    does for wealth): (..., n_states) -> (..., n_states, n_actions)."""

    def _one(s):
        idxs, probs = distribute(grid, s)
        return jnp.zeros(grid.shape[0]).at[idxs].add(probs)

    flat = shares.reshape(-1)
    return jax.vmap(_one)(flat).reshape(*shares.shape, grid.shape[0])


class GPDPAgent:
    """
    A policy GP behind MFAX's agent interface, so that MFAX's evaluation code (sequence.py,
    exploitability.py) runs it unchanged. sequence.py treats an agent with `init_hidden` as recurrent;
    here the hidden state is just the period t (a clock), reset when an episode is done.
    - Observed prices (r, w) are turned back into (z, K): average income L is constant in this model,
      so r / w = alpha / (1 - alpha) * L / K, and then r = e^z alpha (L / K)^(1 - alpha).
    - MFAX's environment loops and best response work on the discrete actions, so each share is put on
      the grid: grid_mode="split" splits it between its two neighbouring grid actions (same mean
      share, but a lottery over consumption), grid_mode="nearest" takes the nearest grid action.
    """

    def __init__(self, env, mf_params, grid_mode: str = "split"):
        p = env.params
        self.mf_params = mf_params  # the policy GP (GPState)
        self.alpha = float(p.cobb_douglas_alpha)
        self.L = float(np.mean(np.asarray(p.discrete_states[1])))  # uniform income marginal
        self.T = int(p.max_steps_in_episode)
        self.grid = np.asarray(p.discrete_actions)
        self.grid_mode = grid_mode

    def init_hidden(self, batch_size: int):
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def shares(self, aggregate_obs, t, mf_params=None):
        gp = self.mf_params if mf_params is None else mf_params
        r, w = aggregate_obs[:, 0], aggregate_obs[:, 1]
        K = self.alpha / (1 - self.alpha) * self.L * w / r
        z = jnp.log(r / (self.alpha * (self.L / K) ** (1 - self.alpha)))
        x = jnp.stack([z, K, (self.T - t).astype(K.dtype)], axis=-1)
        if not uses_time(gp):  # the time-input ablation: the policy GP takes (z, K) only
            x = x[:, :2]
        return jax.nn.sigmoid(gp_predict(gp, x))

    def __call__(self, individual_states, aggregate_obs, hidden_state, done=None, mf_params=None):
        t = hidden_state if done is None else jnp.where(done, 0, hidden_state)
        shares, grid = self.shares(aggregate_obs, t, mf_params), jnp.asarray(self.grid)
        if getattr(self, "grid_mode", "split") == "nearest":
            nearest = jnp.argmin(jnp.abs(shares[..., None] - grid), axis=-1)
            prob_a = jax.nn.one_hot(nearest, grid.shape[0])
        else:
            prob_a = shares_to_grid_probs(shares, grid)
        return prob_a, t + 1


# --- a policy GP with memory: RSPG's information set ---------------------------------------------
# RSPG's network sees the household state and an embedding of a GRU that summarizes the history of
# prices. Here the household state indexes the policy GP's outputs, and its inputs are (z, K) (what
# current prices reveal) plus the hidden state of a GRU taken from a trained RSPG run and frozen.
# No explicit date: like RSPG, the policy can only infer time from the price history.
@struct.dataclass
class GRUFeatures:
    """A frozen RSPG price-history encoder (mfax.utils.nets.base.RecurrentEncoder)."""

    params: dict
    hidden_size: int = struct.field(pytree_node=False)
    embed_size: int = struct.field(pytree_node=False)
    activation: str = struct.field(pytree_node=False)

    def step(self, hidden, obs, done):
        """One GRU update from the observed prices (E, 2): new hidden state (E, hidden_size)."""
        module = RecurrentEncoder(self.hidden_size, self.embed_size, self.activation)
        new_hidden, _ = module.apply({"params": self.params}, hidden, obs, done)
        return new_hidden


def gru_from_run(run_dir) -> GRUFeatures:
    """The GRU of a saved RSPG run (mf_agent_wrapper.pkl), as fixed features."""
    with open(Path(run_dir) / "mf_agent_wrapper.pkl", "rb") as f:
        wrapper = pickle.load(f)
    enc = wrapper.mf_policy.aggregate_encoder
    return GRUFeatures(jax.device_get(wrapper.mf_params["encoder"]), enc.hidden_size, enc.embed_size, enc.activation)


@struct.dataclass
class RecurrentPolicyGP:
    """Policy GP over (z, K, GRU hidden state)."""

    gp: GPState
    gru: GRUFeatures


def observed_prices(aggregate_s):
    """What households observe: (r, w), batched (E, 2)."""
    return jnp.stack([aggregate_s.interest_rate, aggregate_s.wage], axis=-1)


def memory_inputs(env, aggregate_s, hidden, with_zk: bool = True):
    """(z, K, h_1 .. h_d) for a batch of economies, (E, 2 + d); or the hidden state only, (E, d)."""
    if not with_zk:
        return hidden
    return jnp.concatenate([aggregate_features(env, aggregate_s), hidden], axis=-1)


def uses_zk(policy) -> bool:
    """Whether a memory policy GP also takes (z, K), from its number of inputs."""
    return policy.gp.X.shape[-1] != policy.gru.hidden_size


def training_data_memory(env, aggregate_s_seq, hidden_seq, shares, with_zk: bool = True):
    """Policy GP training pairs with memory inputs: like training_data, inputs (z, K, h) or h."""
    n_periods = shares.shape[0]
    X = jax.vmap(lambda a, h: memory_inputs(env, a, h, with_zk))(
        jax.tree.map(lambda x: x[:n_periods], aggregate_s_seq), hidden_seq[:n_periods]
    )
    return X.reshape(-1, X.shape[-1]), logit(shares).reshape(-1, shares.shape[-1])


def predict_shares_memory(env, policy: RecurrentPolicyGP, aggregate_s, hidden):
    return jax.nn.sigmoid(gp_predict(policy.gp, memory_inputs(env, aggregate_s, hidden, uses_zk(policy))))


class GPDPRecurrentAgent(GPDPAgent):
    """
    RecurrentPolicyGP behind MFAX's recurrent-agent interface: its hidden state is the frozen GRU's,
    updated from the observed prices exactly as RSPG's is. (z, K) are recovered from prices as in
    GPDPAgent.
    """

    def __init__(self, env, mf_params, grid_mode: str = "split", hidden_size=None):
        super().__init__(env, mf_params, grid_mode)
        self.hidden_size = mf_params.gru.hidden_size if mf_params is not None else hidden_size

    def init_hidden(self, batch_size: int):
        return jnp.zeros((batch_size, self.hidden_size), dtype=jnp.float32)

    def __call__(self, individual_states, aggregate_obs, hidden_state, done=None, mf_params=None):
        policy = self.mf_params if mf_params is None else mf_params
        if done is None:
            done = jnp.zeros((aggregate_obs.shape[0],), dtype=bool)
        hidden = policy.gru.step(hidden_state, aggregate_obs, done)
        r, w = aggregate_obs[:, 0], aggregate_obs[:, 1]
        K = self.alpha / (1 - self.alpha) * self.L * w / r
        z = jnp.log(r / (self.alpha * (self.L / K) ** (1 - self.alpha)))
        x = jnp.concatenate([jnp.stack([z, K], axis=-1), hidden], axis=-1) if uses_zk(policy) else hidden
        shares, grid = jax.nn.sigmoid(gp_predict(policy.gp, x)), jnp.asarray(self.grid)
        if getattr(self, "grid_mode", "split") == "nearest":
            prob_a = jax.nn.one_hot(jnp.argmin(jnp.abs(shares[..., None] - grid), axis=-1), grid.shape[0])
        else:
            prob_a = shares_to_grid_probs(shares, grid)
        return prob_a, hidden
