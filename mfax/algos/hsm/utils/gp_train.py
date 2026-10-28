"""
SPG / RSPG with a Gaussian-process critic: truncated rollouts, GP tail value.

Objective for one gradient step (per economy e, window of H periods starting at period t_e):

    J_e = sum_x mu_{t_e}(x) * V^e_0(x),    V_t = E_pi[r_t] + gamma * E[V_{t+1}],    V_H = stop_grad(V_GP(x_H))

Restart scheme ("staggered windows"). The 128-period episode is cut into windows of H periods.
Economy slots are assigned to phases k = 0, 1, 2, ...: phase-0 slots start from the environment
reset (uniform mu_0, z = 0), and a phase-k slot continues the economy that a phase-(k-1) slot
simulated at the previous gradient step. Slot weights are gamma^(H k) / n_k, so that

    E[gradient] = sum_k gamma^(H k) E[gradient of window k] ~= gradient of the full-rollout objective,

up to critic error and the omitted phases. This is the actor-critic decomposition of the original
(full 128-period) SPG gradient: the GP tail of window k carries the effect of window-k actions on
the future, and later windows carry the effect of later actions. The training states therefore
follow the same distribution as the evaluation (episodes from the reset), with no burn-in.

RSPG: the GRU state at the start of a window is recomputed with the current parameters from the
stored observation history since the episode start (replay_hidden), with gradients, so the GRU
receives the same gradient as with full backpropagation through time.

With phase_counts = (num_envs,), horizon = max_steps_in_episode and use_gp = False, the train step
is identical to the original spg.py / rspg.py train step (same random numbers, same gradient).
"""

import json
import os
import pickle
import time
from dataclasses import asdict
from typing import Any, NamedTuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState

from mfax.algos.hsm.exploitability import make_exploitability
from mfax.algos.hsm.utils.gp_critic import (
    aggregate_features,
    empty_gp,
    fit_gp,
    gp_inputs,
    gp_predict,
    init_feature_history,
    init_hyper,
    n_base_inputs,
    push_feature_history,
    subsample,
)
from mfax.algos.hsm.utils.gp_rollouts import (
    MFPolicyFn,
    replay_hidden,
    reset_batch,
    select_rows,
    value_rollout,
)
from mfax.algos.hsm.utils.make_act import MFActorWrapper, MFRecurrentActorWrapper
from mfax.algos.hsm.utils.mf_policy_wrappers import (
    MeanFieldContinuousPolicy,
    MeanFieldPolicy,
    RecurrentMeanFieldContinuousPolicy,
    RecurrentMeanFieldPolicy,
)
from mfax.envs import make_env


class EconState(NamedTuple):
    """What each economy slot carries from one gradient step to the next."""

    aggregate_s: Any
    aggregate_obs: jax.Array
    obs_hist: jax.Array  # (E, L, obs_dim) observations since the episode start, right-aligned
    feat_hist: jax.Array  # (E, n_lags + 1, 2) GP features, newest first


class Transition(NamedTuple):
    aggregate_obs: jax.Array
    aggregate_s: Any
    mat_r: jax.Array
    prob_a: jax.Array
    gp_x: jax.Array


class PhaseLayout(NamedTuple):
    slot_phase: np.ndarray  # (E,) phase of each slot
    source: np.ndarray  # (E,) slot (at the previous step) a slot continues from; -1 = reset
    weights: np.ndarray  # (E,) gamma^(H k) / n_k
    n_phases: int


def phase_layout(phase_counts, horizon: int, gamma: float) -> PhaseLayout:
    counts = list(phase_counts)
    assert all(counts[k] <= counts[k - 1] for k in range(1, len(counts))), (
        f"phase_counts must be non-increasing (a phase-k slot continues a phase-(k-1) slot): {counts}"
    )
    starts = np.cumsum([0] + counts[:-1])
    slot_phase, source = [], []
    for k, n in enumerate(counts):
        for j in range(n):
            slot_phase.append(k)
            source.append(-1 if k == 0 else int(starts[k - 1] + j))
    slot_phase = np.array(slot_phase)
    weights = np.array([gamma ** (horizon * k) / counts[k] for k in slot_phase])
    return PhaseLayout(slot_phase, np.array(source), weights, len(counts))


def _strong(tree):
    """Drop weak types so that scan carries keep a fixed type."""
    return jax.tree.map(lambda x: jnp.asarray(x).astype(jnp.asarray(x).dtype), tree)


def build_policy(args, env, recurrent: bool):
    """The same networks as spg.py / rspg.py."""
    discrete = isinstance(env.action_space, gymnax.environments.spaces.Discrete)
    if recurrent:
        encoder_kwargs = dict(
            hidden_size=getattr(args, "gru_hidden_size", 128),
            embed_size=128,
            activation=args.activation,
        )
        if discrete:
            policy_kwargs = dict(
                n_actions=env.n_actions,
                hidden_layer_sizes=(128, 128, 128),
                activation=args.activation,
                state_type=args.state_type,
                num_states=env.n_states,
            )
            return RecurrentMeanFieldPolicy(
                state_type=args.state_type,
                num_states=env.n_states,
                encoder_kwargs=encoder_kwargs,
                policy_kwargs=policy_kwargs,
            )
        policy_kwargs = dict(
            action_dim=env.action_space.shape[-1],
            action_range=(env.action_space.low, env.action_space.high),
            hidden_layer_sizes=(128, 128, 128),
            activation=args.activation,
            state_type=args.state_type,
            num_states=env.n_states,
        )
        return RecurrentMeanFieldContinuousPolicy(
            state_type=args.state_type,
            num_states=env.n_states,
            actions=env.params.discrete_actions,
            encoder_kwargs=encoder_kwargs,
            policy_kwargs=policy_kwargs,
        )
    if discrete:
        policy_kwargs = dict(
            n_actions=env.n_actions,
            hidden_layer_sizes=(128, 128, 128),
            activation=args.activation,
        )
        return MeanFieldPolicy(
            state_type=args.state_type,
            num_states=env.n_states,
            policy_kwargs=policy_kwargs,
        )
    policy_kwargs = dict(
        action_dim=env.action_space.shape[-1],
        action_range=(env.action_space.low, env.action_space.high),
        hidden_layer_sizes=(128, 128, 128),
        activation=args.activation,
        state_type=args.state_type,
        num_states=env.n_states,
    )
    return MeanFieldContinuousPolicy(
        state_type=args.state_type,
        num_states=env.n_states,
        actions=env.params.discrete_actions,
        policy_kwargs=policy_kwargs,
    )


def init_actor(args, env, mf_policy_net, individual_states, recurrent: bool):
    """Same initialisation (and random numbers) as spg.py / rspg.py."""
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_actor_params, rng_action = jax.random.split(rng, 3)
    init_obs = jnp.ones((args.num_envs, env.obs_dim), dtype=jnp.float32)
    states = env.normalize_individual_s(individual_states, args.normalize_states)
    obs = env.normalize_obs(init_obs, args.normalize_obs)
    if recurrent:
        init_hidden = mf_policy_net.init_hidden(
            args.num_envs, mf_policy_net.aggregate_encoder.hidden_size
        )
        init_done = jnp.zeros((args.num_envs,), dtype=jnp.bool_)
        actor_params = mf_policy_net.init(
            rng_actor_params, states, obs, init_hidden, init_done, rng_action
        )
    else:
        actor_params = mf_policy_net.init(rng_actor_params, states, obs, rng_action)

    if args.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(
                learning_rate=optax.linear_schedule(
                    args.lr, args.lr * 0.1, args.num_iterations
                ),
                eps=1e-8,
            ),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-8),
        )
    return TrainState.create(apply_fn=(), params=actor_params, tx=tx), rng


def make_loss(args, env, policy_fn: MFPolicyFn, layout: PhaseLayout):
    E = args.num_envs
    H = args.horizon
    T = env.params.max_steps_in_episode
    L = H * (layout.n_phases - 1)
    is_reset = jnp.asarray(layout.slot_phase == 0)

    def _gp_x(feat_hist, aggregate_s):
        return gp_inputs(feat_hist, T - aggregate_s.time if args.time_input else None)
    src = jnp.asarray(np.maximum(layout.source, 0))
    n_valid = jnp.asarray(H * layout.slot_phase)
    weights = jnp.asarray(layout.weights, dtype=jnp.float32)

    def _loss(params, gp, econ: EconState, rng):
        # --- starting state of each slot: reset, or continue the economy of its source slot ---
        rng, reset_rng, loop_rng = jax.random.split(rng, 3)
        reset_rng = jax.random.split(reset_rng, E)
        reset_obs, reset_s = jax.vmap(env.mf_reset, in_axes=(0))(reset_rng)
        carried = jax.tree.map(lambda x: x[src], econ)
        aggregate_s = select_rows(is_reset, reset_s, carried.aggregate_s)
        aggregate_obs = select_rows(is_reset, reset_obs, carried.aggregate_obs)
        obs_hist = select_rows(
            is_reset, jnp.zeros_like(carried.obs_hist), carried.obs_hist
        )
        feat_hist = select_rows(
            is_reset,
            init_feature_history(aggregate_features(env, reset_s), args.n_lags),
            carried.feat_hist,
        )
        if policy_fn.recurrent:
            hidden = replay_hidden(policy_fn, params, obs_hist, n_valid)
        else:
            hidden = policy_fn.init_hidden(E)
        done = jnp.zeros((E,), dtype=jnp.bool_)

        # --- simulate H periods ---
        def _mf_transition(runner_state, unused):
            aggregate_s, aggregate_obs, hidden, done, feat_hist, rng = runner_state
            rng, rng_env, rng_next = jax.random.split(rng, 3)
            prob_a, next_hidden = policy_fn(params, hidden, aggregate_obs, done)
            rng_env = jax.random.split(rng_env, E)
            (
                _,
                next_aggregate_obs,
                _,
                next_aggregate_s,
                mat_r,
                terminated,
                truncated,
                _,
            ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0))(rng_env, aggregate_s, prob_a)
            next_feat_hist = push_feature_history(
                feat_hist, aggregate_features(env, next_aggregate_s)
            )
            transition = Transition(
                aggregate_obs, aggregate_s, mat_r, prob_a, _gp_x(feat_hist, aggregate_s)
            )
            runner_state = (
                next_aggregate_s,
                next_aggregate_obs,
                next_hidden,
                jnp.logical_or(terminated, truncated),
                next_feat_hist,
                rng_next,
            )
            return runner_state, transition

        final, traj = jax.lax.scan(
            _mf_transition,
            (aggregate_s, aggregate_obs, hidden, done, feat_hist, loop_rng),
            None,
            H,
        )
        final_s, final_obs, _, _, final_feat_hist, _ = final

        # --- tail value: GP critic, held fixed within the step; zero if the episode has ended ---
        if args.use_gp:
            tail = jax.lax.stop_gradient(gp_predict(gp, _gp_x(final_feat_hist, final_s)))
            tail = jnp.where((final_s.time >= T)[:, None], 0.0, tail)
        else:
            tail = jnp.zeros((E, env.n_states))

        # --- backward recursion (policy evaluation) ---
        def _get_expected_return(disc_rewards, transition):
            expected = jax.vmap(env.mf_expected_value, in_axes=(0, 0, 0))
            disc_rewards = jnp.sum(
                transition.prob_a * transition.mat_r, axis=-1
            ) + args.discount_factor * expected(
                disc_rewards, transition.prob_a, transition.aggregate_s
            )
            return disc_rewards, disc_rewards

        disc_rewards, V = jax.lax.scan(
            _get_expected_return, tail, traj, reverse=True, unroll=16
        )
        per_econ = jnp.sum(disc_rewards * traj.aggregate_s.mu[0], axis=-1)
        loss = -jnp.sum(weights * per_econ)

        # --- carry each economy forward ---
        if L > 0:
            new_hist = jnp.concatenate(
                [obs_hist, jnp.swapaxes(traj.aggregate_obs, 0, 1)], axis=1
            )[:, -L:]
        else:
            new_hist = obs_hist
        new_econ = _strong(
            jax.lax.stop_gradient(
                EconState(final_s, final_obs, new_hist, final_feat_hist)
            )
        )
        return loss, (new_econ, traj.gp_x, jax.lax.stop_gradient(V), per_econ)

    return _loss


def initial_econ(args, env, policy_fn: MFPolicyFn, layout: PhaseLayout, key):
    """Placeholder carried state (all slots reset); filled by `fill_pipeline` before training."""
    s, obs, _, fh = reset_batch(env, key, args.num_envs, args.n_lags, policy_fn)
    L = args.horizon * (layout.n_phases - 1)
    obs_hist = jnp.zeros((args.num_envs, L, obs.shape[-1]), dtype=jnp.float32)
    return _strong(EconState(s, obs, obs_hist, fh))


def make_train_step(args, env, policy_fn: MFPolicyFn, layout: PhaseLayout):
    _loss = make_loss(args, env, policy_fn, layout)

    def _train_step(runner_state, unused):
        actor_ts, gp, econ, rng = runner_state
        rng, _rng = jax.random.split(rng)
        (loss, (new_econ, gp_x, V, _)), grads = jax.value_and_grad(
            _loss, has_aux=True
        )(actor_ts.params, gp, econ, _rng)
        actor_ts = actor_ts.apply_gradients(grads=grads)
        outputs = (loss, gp_x, V) if args.use_gp else (loss,)
        return (actor_ts, gp, new_econ, rng), outputs

    return _train_step


def make_fill_pipeline(args, env, policy_fn: MFPolicyFn, layout: PhaseLayout):
    """Advance all slots once without updating the policy (used to fill phases > 0 at the start)."""
    _loss = make_loss(args, env, policy_fn, layout)

    @jax.jit
    def _fill(params, gp, econ, rng):
        _, (new_econ, _, _, _) = _loss(params, gp, econ, rng)
        return new_econ

    return _fill


def warmstart_targets(args, env, policy_fn: MFPolicyFn, params, key):
    """
    First critic, from paths from the reset under the given policy.
    - Infinite-horizon critic: long paths with a zero tail after keep + extra periods; the first
      `keep` periods give targets whose missing tail is below gamma^extra.
    - Time-aware critic: whole episodes (T periods, terminal reward at T); every period is a
      target, with the time left as an input.
    """
    k_reset, k_roll = jax.random.split(key)
    s, obs, h, fh = reset_batch(env, k_reset, args.warmstart_envs, args.n_lags, policy_fn)
    T = env.params.max_steps_in_episode
    if args.time_input:
        length, keep = T, T
    else:
        length, keep = args.warmstart_keep + args.warmstart_extra, args.warmstart_keep
    V, X, _ = value_rollout(
        env,
        policy_fn,
        params,
        args.discount_factor,
        s,
        obs,
        h,
        fh,
        k_roll,
        length=length,
        return_all=True,
        finite=args.time_input,
        time_input=args.time_input,
    )
    return X[:keep].reshape(-1, X.shape[-1]), V[:keep].reshape(-1, V.shape[-1])


def _save_pkl(obj, directory, filename):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "wb") as f:
        pickle.dump(obj, f)


def _save_checkpoint(directory, args, mf_agent_wrapper, gp):
    """Agent, critic and arguments, in the same layout as a finished run."""
    _save_pkl(mf_agent_wrapper, directory, "mf_agent_wrapper.pkl")
    _save_pkl(jax.device_get(gp), directory, "gp_state.pkl")
    with open(os.path.join(directory, "args.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)


def train(args, recurrent: bool):
    """Full training run. Returns (actor_ts, gp, log)."""
    if args.state_type == "indices" and args.normalize_states:
        args.normalize_states = False
        print(
            "Normalization of individual states is not supported for indices, setting normalize_states to False"
        )
    assert args.task == "endogenous", (
        "GP features (z, K) are implemented for the endogenous (macro) environment only"
    )
    assert args.num_envs == sum(args.phase_counts), (
        f"num_envs ({args.num_envs}) must equal sum(phase_counts) ({sum(args.phase_counts)})"
    )
    assert args.num_iterations % args.refit_every == 0
    assert args.eval_frequency % args.refit_every == 0
    assert args.fast_refit_steps % args.refit_every == 0

    out_dir = f"runs/{args.task}/{args.algo}" + (
        f"/{args.run_name}" if args.run_name else ""
    )
    if args.save:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "args.json"), "w") as f:
            json.dump(asdict(args), f, indent=2)

    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type=f"{args.task}/{args.algo}",
        )

    # --- environment, policy, layout ---
    env = make_env("pushforward/" + args.task, common_noise=args.common_noise)
    assert args.horizon * len(args.phase_counts) <= env.params.max_steps_in_episode, (
        "windows must fit in the episode"
    )
    individual_states = (
        jnp.arange(env.n_states) if args.state_type == "indices" else env.params.states
    )
    mf_policy_net = build_policy(args, env, recurrent)
    policy_fn = MFPolicyFn(
        env,
        mf_policy_net,
        individual_states,
        args.normalize_obs,
        args.normalize_states,
        recurrent,
    )
    layout = phase_layout(args.phase_counts, args.horizon, args.discount_factor)
    actor_ts, rng = init_actor(args, env, mf_policy_net, individual_states, recurrent)
    n_base = n_base_inputs(args.time_input)
    n_inputs = n_base + 2 * args.n_lags

    # --- exploitability (fixed evaluation key: common random numbers across runs) ---
    wrapper_cls = MFRecurrentActorWrapper if recurrent else MFActorWrapper
    mf_agent_wrapper = wrapper_cls(
        mf_policy_net,
        actor_ts.params,
        env.normalize_obs,
        args.normalize_obs,
        env.normalize_individual_s,
        args.normalize_states,
    )
    exploitability = make_exploitability(
        env=env,
        agent=mf_agent_wrapper,
        state_type=args.state_type,
        gamma=args.discount_factor,
        num_envs=args.eval_envs,
        max_steps_in_episode=env.params.max_steps_in_episode,
    )
    eval_key = jax.random.PRNGKey(args.eval_seed)

    log = dict(curve=[], timing=dict(chunks=[], chunk_steps=[], refits=[], evals=[]), gp=[])

    def _evaluate(iteration, elapsed):
        t0 = time.perf_counter()
        res = exploitability(eval_key, actor_ts.params)
        expl = float(res.exploitability.exploitability)
        pol = float(res.exploitability.mean_policy_return)
        br = float(res.exploitability.mean_br_return)
        log["timing"]["evals"].append(time.perf_counter() - t0)
        log["curve"].append(
            dict(
                iteration=iteration,
                train_time=elapsed,
                exploitability=expl,
                policy_return=pol,
                br_return=br,
            )
        )
        if args.log:
            wandb.log(
                {
                    f"{args.task}/iteration": float(iteration),
                    f"{args.task}/train_time": float(elapsed),
                    f"{args.task}/exploitability": expl,
                    f"{args.task}/policy_disc_return": pol,
                }
            )
        print(
            f"Iteration: {iteration}, Train Time: {elapsed:.1f}, Exploitability: {expl:.4f}, Policy return: {pol:.4f}",
            flush=True,
        )

    # --- initial evaluation (not timed, as in spg.py / rspg.py) ---
    if args.evaluate:
        _evaluate(0, 0.0)

    elapsed = 0.0
    gp = empty_gp(args.n_gp, n_inputs, env.n_states)

    # --- fill the staggered pipeline (phases > 0) under the initial policy ---
    t0 = time.perf_counter()
    rng, k_econ = jax.random.split(rng)
    econ = initial_econ(args, env, policy_fn, layout, k_econ)
    if layout.n_phases > 1:
        fill = make_fill_pipeline(args, env, policy_fn, layout)
        for _ in range(layout.n_phases - 1):
            rng, k = jax.random.split(rng)
            econ = fill(actor_ts.params, gp, econ, k)
        jax.block_until_ready(econ)
    log["timing"]["fill"] = time.perf_counter() - t0
    elapsed += log["timing"]["fill"]

    # --- first critic ---
    if args.use_gp:
        t0 = time.perf_counter()
        rng, k_ws, k_sub = jax.random.split(rng, 3)
        X, Y = warmstart_targets(args, env, policy_fn, actor_ts.params, k_ws)
        Xs, Ys = subsample(k_sub, X, Y, args.n_gp)
        gp, nll = fit_gp(
            Xs, Ys, init_hyper(n_inputs, n_base), args.gp_steps_first, args.gp_lr
        )
        jax.block_until_ready(gp)
        log["timing"]["warmstart"] = time.perf_counter() - t0
        elapsed += log["timing"]["warmstart"]
        log["gp"].append(
            dict(
                iteration=0,
                nll=float(nll),
                ell=np.exp(np.asarray(gp.hyp.log_ell)).tolist(),
                sf=float(np.exp(gp.hyp.log_sf)),
                sn=float(np.exp(gp.hyp.log_sn)),
            )
        )
        print(
            f"First GP fit: {log['timing']['warmstart']:.1f}s, NLML {float(nll):.4f}, "
            f"length scales {np.round(np.exp(np.asarray(gp.hyp.log_ell)), 3).tolist()}",
            flush=True,
        )

    # --- training ---
    train_step = make_train_step(args, env, policy_fn, layout)

    @jax.jit
    def train_chunk(runner_state):
        return jax.lax.scan(train_step, runner_state, None, args.refit_every)

    @jax.jit
    def train_one(runner_state):
        return jax.lax.scan(train_step, runner_state, None, 1)

    @jax.jit
    def _subsample(key, gp_x, V):
        return subsample(
            key, gp_x.reshape(-1, gp_x.shape[-1]), V.reshape(-1, V.shape[-1]), args.n_gp
        )

    runner_state = (actor_ts, gp, econ, rng)
    iteration = 0
    recent = []  # targets of the most recent steps, for the fast refits
    while iteration < args.num_iterations:
        # --- early in training the policy moves fast: refit the critic after every step ---
        fast = args.use_gp and iteration < args.fast_refit_steps
        n_steps = 1 if fast else args.refit_every
        t0 = time.perf_counter()
        runner_state, outputs = (train_one if fast else train_chunk)(runner_state)
        jax.block_until_ready(runner_state)
        t_chunk = time.perf_counter() - t0
        log["timing"]["chunks"].append(t_chunk)
        log["timing"]["chunk_steps"].append(n_steps)
        elapsed += t_chunk
        iteration += n_steps
        actor_ts, gp, econ, rng = runner_state

        if args.use_gp:
            t0 = time.perf_counter()
            _, gp_x, V = outputs
            if fast:
                recent = (recent + [(gp_x, V)])[-args.fast_refit_window :]
                gp_x = jnp.concatenate([g for g, _ in recent])
                V = jnp.concatenate([v for _, v in recent])
            rng, k_sub = jax.random.split(rng)
            Xs, Ys = _subsample(k_sub, gp_x, V)
            gp, nll = fit_gp(Xs, Ys, gp.hyp, args.gp_steps_refit, args.gp_lr)
            jax.block_until_ready(gp)
            t_refit = time.perf_counter() - t0
            log["timing"]["refits"].append(t_refit)
            elapsed += t_refit
            log["gp"].append(
                dict(
                    iteration=iteration,
                    nll=float(nll),
                    ell=np.exp(np.asarray(gp.hyp.log_ell)).tolist(),
                    sf=float(np.exp(gp.hyp.log_sf)),
                    sn=float(np.exp(gp.hyp.log_sn)),
                )
            )
            runner_state = (actor_ts, gp, econ, rng)

        if args.evaluate and iteration % args.eval_frequency == 0:
            _evaluate(iteration, elapsed)

        # --- checkpoints for validating the critic during training (not timed) ---
        if (
            args.save
            and args.checkpoint_every
            and iteration % args.checkpoint_every == 0
            and iteration < args.num_iterations
        ):
            _save_checkpoint(
                os.path.join(out_dir, f"ckpt_{iteration}"),
                args,
                wrapper_cls(
                    mf_policy_net,
                    actor_ts.params,
                    env.normalize_obs,
                    args.normalize_obs,
                    env.normalize_individual_s,
                    args.normalize_states,
                ),
                gp,
            )

    log["timing"]["total_train_time"] = elapsed
    chunks = log["timing"]["chunks"]
    # --- time per step, leaving out the first chunk of each length (it includes compilation) ---
    per_step, seen = [], set()
    for t, n in zip(chunks, log["timing"]["chunk_steps"]):
        if n in seen:
            per_step.append(t / n)
        seen.add(n)
    log["timing"]["step_time_steady"] = float(
        np.mean(per_step) if per_step else chunks[0] / log["timing"]["chunk_steps"][0]
    )
    log["timing"]["first_chunk_incl_compile"] = chunks[0]
    if log["timing"]["refits"]:
        log["timing"]["refit_time_steady"] = float(
            np.mean(log["timing"]["refits"][1:] or log["timing"]["refits"])
        )
    print(
        f"Train time {elapsed:.1f}s, steady step time {log['timing']['step_time_steady']:.3f}s",
        flush=True,
    )

    # --- save ---
    if args.save:
        mf_agent_wrapper = wrapper_cls(
            mf_policy_net,
            actor_ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        _save_pkl(mf_agent_wrapper, out_dir, "mf_agent_wrapper.pkl")
        _save_pkl(jax.device_get(gp), out_dir, "gp_state.pkl")
        with open(os.path.join(out_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"Saved to {out_dir}")

    if args.log:
        wandb.finish()
    return actor_ts, gp, log
