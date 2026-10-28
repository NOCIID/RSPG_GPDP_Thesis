"""
GPDP for the endogenous mean-field game (build step 4 in FINDINGS.md): no gradient descent and no
policy network. The population's policy is a policy GP of continuous consumption shares.

One iteration:
    1. forward:  simulate N economies (the stations), t = 0 .. T-1, under the population's policy
    2. backward: best responses at the stations with one V-GP per period (gpdp_backward.py); they
                 average over tomorrow's shock, so they do not use the realized future
    3. policy:   fit the policy GP on blended targets (1 - alpha) * population share + alpha * best share
                 (gpdp_policy.py); it becomes the population's policy
Each iteration also evaluates the population's policy before the update (on fixed fresh economies):
how much a household gains by deviating, honestly (following the V-GPs) and clairvoyantly, and
MFAX's own exploitability (grid best response) through GPDPAgent, as gp_train.py evaluates RSPG-GP.
"""

import json
import os
import pickle
import time
from dataclasses import asdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mfax.algos.hsm.best_response import br
from mfax.algos.hsm.evaluate import evaluate_given_sequence
from mfax.algos.hsm.exploitability import make_exploitability
from mfax.algos.hsm.utils import gpdp_backward as gb
from mfax.algos.hsm.utils import gpdp_continuous as gc
from mfax.algos.hsm.utils import gpdp_lom as glom
from mfax.algos.hsm.utils import gpdp_policy as gpol
from mfax.algos.hsm.utils.gp_critic import GPState, aggregate_features, init_hyper
from mfax.envs import make_env
from mfax.envs.pushforward.base import PushforwardMFSequence


# --- the population's policy: a constant share (the start), a policy GP (GPState), or a policy GP
#     with memory (RecurrentPolicyGP: inputs (z, K, GRU hidden state)) ---
def policy_shares(env, policy, aggregate_s, hidden=None):
    """Shares of every state for a batch of economies (leading axis E): (E, n_states)."""
    if isinstance(policy, gpol.RecurrentPolicyGP):
        return gpol.predict_shares_memory(env, policy, aggregate_s, hidden)
    if isinstance(policy, GPState):
        return gpol.predict_shares(env, policy, aggregate_s)
    return jnp.full((aggregate_s.time.shape[0], env.n_states), policy)


@partial(jax.jit, static_argnames=("env", "n_econ"))
def forward(env, policy, key, n_econ, gru=None):
    """
    Simulate n_econ economies from the reset under the population's policy (a constant share or a
    policy GP). Returns (aggregate states (T, E), shares (T, E, n_states)); the last period's share
    is unused (terminal reward).
    With memory (a RecurrentPolicyGP, or a frozen GRU passed as `gru`), the GRU is run on each
    economy's observed prices, as RSPG's is, and the hidden states (T, E, d) are returned as well.
    """
    T = env.params.max_steps_in_episode
    if isinstance(policy, gpol.RecurrentPolicyGP):
        gru = policy.gru
    k_reset, k_steps = jax.random.split(key)
    _, aggregate_s = jax.vmap(env.mf_reset)(jax.random.split(k_reset, n_econ))
    no_done = jnp.zeros((n_econ,), dtype=bool)

    def _memory(hidden, aggregate_s):
        if gru is None:
            return hidden
        return gru.step(hidden, gpol.observed_prices(aggregate_s), no_done)

    def _step(carry, k):
        aggregate_s, hidden = carry
        hidden = _memory(hidden, aggregate_s)  # after observing this period's prices, as RSPG
        shares = policy_shares(env, policy, aggregate_s, hidden)
        next_aggregate_s, _, _ = jax.vmap(
            lambda kk, a, s: gc.step(env, kk, a, s)
        )(jax.random.split(k, n_econ), aggregate_s, shares)
        return (next_aggregate_s, hidden), (aggregate_s, shares, hidden)

    hidden0 = jnp.zeros((n_econ, gru.hidden_size if gru is not None else 1), jnp.float32)
    (last, hidden), (seq, shares, hiddens) = jax.lax.scan(
        _step, (aggregate_s, hidden0), jax.random.split(k_steps, T - 1)
    )
    hidden_last = _memory(hidden, last)
    seq = jax.tree.map(lambda x, y: jnp.concatenate([x, y[None]], axis=0), seq, last)
    shares = jnp.concatenate([shares, policy_shares(env, policy, last, hidden_last)[None]], axis=0)
    if gru is None:
        return seq, shares
    return seq, shares, jnp.concatenate([hiddens, hidden_last[None]], axis=0)


@partial(jax.jit, static_argnames=("env", "n_econ"))
def forward_grid(env, policy, key, n_econ):
    """
    Simulate n_econ economies with the population's policy played on the grid actions (each share
    split between its two neighbouring grid actions, as GPDPAgent does), stepping with MFAX's own
    env.mf_step. Returns (aggregate states (T, E), effective mean shares (T, E, n_states), and the
    PushforwardMFSequence that MFAX's evaluate_given_sequence and br take).
    """
    T = env.params.max_steps_in_episode
    grid = env.params.discrete_actions
    k_reset, k_steps = jax.random.split(key)
    obs, aggregate_s = jax.vmap(env.mf_reset)(jax.random.split(k_reset, n_econ))

    def _step(carry, k):
        aggregate_s, obs = carry
        prob_a = gpol.shares_to_grid_probs(policy_shares(env, policy, aggregate_s), grid)
        _, next_obs, _, next_aggregate_s, mat_r, terminated, truncated, _ = jax.vmap(
            env.mf_step
        )(jax.random.split(k, n_econ), aggregate_s, prob_a)
        return (next_aggregate_s, next_obs), (aggregate_s, obs, prob_a, mat_r, terminated, truncated)

    _, (seq, obs_seq, prob_a, mat_r, terminated, truncated) = jax.lax.scan(
        _step, (aggregate_s, obs), jax.random.split(k_steps, T)
    )
    traj = PushforwardMFSequence(
        aggregate_s=seq,
        aggregate_terminated=terminated,
        aggregate_truncated=truncated,
        aggregate_obs=obs_seq,
        aggregate_hidden=None,
        prob_a=prob_a,
        mat_r=mat_r,
    )
    return seq, (prob_a * grid).sum(-1), traj


def _mu_weighted_mean(values, aggregate_s_seq, t):
    """Population-weighted mean over individual states at period t, averaged over economies."""
    mu = np.asarray(aggregate_s_seq.mu[t])
    return float((np.asarray(values) * mu / mu.sum(-1, keepdims=True)).sum(-1).mean())


def _ce(v_hi, v_lo, sigma):
    """Consumption-equivalent gain of value v_hi over v_lo (CRRA)."""
    return (v_hi / v_lo) ** (1 / (1 - sigma)) - 1


def train(args):
    env = make_env("pushforward/" + args.task, common_noise=args.common_noise)
    assert args.task == "endogenous", "GP features (z, K) are implemented for the endogenous model only"
    gamma, sigma = args.discount_factor, env.params.sigma
    T = env.params.max_steps_in_episode
    out_dir = f"runs/{args.task}/{args.algo}" + (f"/{args.run_name}" if args.run_name else "")
    if args.save:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "args.json"), "w") as f:
            json.dump(asdict(args), f, indent=2)

    # --- evaluation (fixed economies and keys across iterations, as in gp_train.py) ---
    eval_key = jax.random.PRNGKey(args.eval_seed)
    # --- optional memory: a frozen GRU from a trained RSPG run (RSPG's information set) ---
    memory_run = getattr(args, "policy_memory_run", "")
    gru = gpol.gru_from_run(memory_run) if memory_run else None
    if gru is not None:
        assert not getattr(args, "grid_actions", False), "memory is implemented for continuous shares"
        assert not getattr(args, "policy_time", True), "with memory, run with --no-policy-time"
        agent = gpol.GPDPRecurrentAgent(env, None, hidden_size=gru.hidden_size)
    else:
        agent = gpol.GPDPAgent(env, None)
    mfax_exploitability = None  # compiled on the first policy GP (its shapes are fixed afterwards)
    path_br = jax.jit(lambda a: gc.path_best_response(env, a, gamma)[0])
    path_value = jax.jit(lambda a, s: gc.path_evaluate(env, a, s, gamma)[0])

    grid_actions = getattr(args, "grid_actions", False)

    def simulate(policy, key, n_econ):
        """(stations, effective shares, MFAX sequence or None, GRU hidden states or None)."""
        if grid_actions:
            return (*forward_grid(env, policy, key, n_econ), None)
        out = forward(env, policy, key, n_econ, gru)
        return out[0], out[1], None, (out[2] if gru is not None else None)

    def evaluate(policy, gps, lom=None):
        """
        Gains from deviating from `policy` (% of consumption), on fixed fresh economies. The honest
        deviation follows the V-GPs (continuous shares, or grid actions with --grid-actions); the
        clairvoyant one knows the realized shocks (continuous shares; with --grid-actions also MFAX's br).
        With a law of motion, also its accuracy on these economies (it was fitted on the stations).
        """
        seq, shares, traj, _ = simulate(policy, eval_key, args.eval_envs)
        mu0 = np.asarray(seq.mu[0])
        w = lambda v: float((np.asarray(v) * mu0).sum(-1).mean())  # noqa: E731
        if grid_actions:  # the population plays a mix of grid actions: MFAX's own evaluation
            v_pol = w(evaluate_given_sequence(env, traj, gamma=gamma)[0])
        else:
            v_pol = w(path_value(seq, shares))
        honest = gb.greedy_shares(env, seq, gps, gamma, n_nodes=args.n_nodes, grid_only=grid_actions)
        v_honest = w(path_value(seq, honest))
        v_clair = w(path_br(seq))
        extra = {}
        if grid_actions:
            extra["gain_clairvoyant_grid_pct"] = 100 * _ce(w(br(env, traj, gamma)[0]), v_pol, sigma)
        if lom is not None:
            X_e, y_e = (np.asarray(a, np.float64) for a in glom.transitions(env, seq))
            m, v = glom.predict_next_K(lom, X_e)
            err = X_e[:, 1] + y_e[:, 0] - m
            dK = y_e[:, 0]
            extra["law_of_motion"] = dict(
                r2_of_delta_K=float(1 - np.sum(err**2) / np.sum((dK - dK.mean()) ** 2)),
                max_rel_err_pct=float(100 * np.max(np.abs(err) / (X_e[:, 1] + y_e[:, 0]))),
                zscore_std=float(np.std(err / np.sqrt(v))),
            )
        return dict(
            policy_value=v_pol,
            gain_honest_pct=100 * _ce(v_honest, v_pol, sigma),
            gain_clairvoyant_pct=100 * _ce(v_clair, v_pol, sigma),
            **extra,
            K_mean_at=dict(
                (str(t), float(np.asarray(aggregate_features(env, jax.tree.map(lambda x: x[t], seq)))[:, 1].mean()))
                for t in (25, 50, 100)
            ),
            mean_share_at=dict((str(t), _mu_weighted_mean(shares[t], seq, t)) for t in (0, 64, 120)),
        )

    # --- starting policy ---
    policy = jnp.float32(args.start_share)
    policy_time = getattr(args, "policy_time", True)
    memory_zk = getattr(args, "policy_memory_zk", True)
    n_policy_inputs = (gru.hidden_size + 2 * memory_zk) if gru is not None else (3 if policy_time else 2)
    hyp = init_hyper(n_policy_inputs, n_policy_inputs)
    single_vgp = getattr(args, "single_vgp", False)
    vgp_hyp = None  # the single value GP's hyperparameters, carried across iterations
    # --- optional law of motion for K (gpdp_lom.py) ---
    law_of_motion = getattr(args, "law_of_motion", False)
    lom_paths = getattr(args, "lom_support_paths", 0)
    lom = None
    if law_of_motion:
        assert single_vgp and not grid_actions and gru is None, "law of motion: --single-vgp, continuous, no memory"
        assert lom_paths == 0 or args.alpha == 1.0, "support paths from the law of motion need --alpha 1"
    key = jax.random.PRNGKey(args.seed)
    log = dict(iterations=[], train_time=0.0)
    for it in range(args.num_iterations):
        t0 = time.perf_counter()
        key, k_fwd = jax.random.split(key)

        # --- 1. forward: the stations ---
        stations, pop_shares, _, hidden = simulate(policy, k_fwd, args.num_stations)
        # --- 2. backward: best responses and V-GPs ---
        if law_of_motion:  # K' from a GP law of motion, closed-form expectation over (z', K')
            # cold start every iteration: warm starting from the previous iteration's hyperparameters got
            # stuck (iteration 0's constant share has no end-of-horizon fall in K, so its time length
            # scale is at the bound, and 100 steps did not bring it back; NLML 1210 vs -4.2 cold)
            X_lom, y_lom = glom.transitions(env, stations)
            lom, lom_nll = glom.fit_law_of_motion(X_lom, y_lom, None, args.lom_steps, args.gp_lr)
            Z, Kc, L = glom.support_from_paths(env, stations)
            template = jax.tree.map(lambda x: x[0], stations)
            if lom_paths > 0:  # support points dreamt by the law of motion, no push-forward of mu
                Z, Kc = glom.support_from_law_of_motion(
                    env, lom, float(Z[0, 0]), float(Kc[0, 0]), seed=args.seed * 100003 + it, n_paths=lom_paths
                )
                template = jax.tree.map(lambda x: jnp.repeat(x[:1], lom_paths, axis=0), template)
                L = jnp.repeat(L[:1], lom_paths)
            gps, best, _, vgp_hyp = glom.backward_pass_single(
                env, template, Z, Kc, L, lom, gamma, n_nodes=args.n_nodes, n_iter=args.search_iter,
                hyp_init=vgp_hyp, hyp_steps_first=args.vgp_steps_first, hyp_steps=args.vgp_hyp_steps,
                n_hyp_cols=args.vgp_hyp_cols, gp_lr=args.gp_lr,
            )
        elif single_vgp:  # one value GP over (z, K, T - t) for all periods
            gps, best, _, vgp_hyp = gb.backward_pass_single(
                env, stations, gamma, n_nodes=args.n_nodes, n_iter=args.search_iter,
                hyp_init=vgp_hyp, hyp_steps_first=args.vgp_steps_first, hyp_steps=args.vgp_hyp_steps,
                n_hyp_cols=args.vgp_hyp_cols, gp_lr=args.gp_lr,
                grid_only=grid_actions,
            )
        else:  # one value GP per period
            gps, best, _, _ = gb.backward_pass(
                env, stations, gamma, n_nodes=args.n_nodes, n_iter=args.search_iter,
                gp_steps_first=args.vgp_steps_first, gp_steps=args.vgp_steps, gp_lr=args.gp_lr,
                grid_only=grid_actions,
            )
        # --- 3. policy: blended targets ---
        if law_of_motion and lom_paths > 0:
            targets = best  # alpha = 1: the population's shares at the support points are not needed
        else:
            targets = (1 - args.alpha) * pop_shares[: T - 1] + args.alpha * best
        if gru is not None:
            X, Y = gpol.training_data_memory(env, stations, hidden, targets, with_zk=memory_zk)
        elif law_of_motion:
            X, Y = glom.policy_training_data(Z, Kc, targets, T)
        else:
            X, Y = gpol.training_data(env, stations, targets, use_time=policy_time)
        policy_kept = False
        if bool(jnp.all(jnp.isfinite(Y))):
            new_gp, nll = gb.fit_gp_bounded(
                X, Y, hyp, args.pgp_steps_first if it == 0 else args.pgp_steps, args.gp_lr
            )
            new_policy = gpol.RecurrentPolicyGP(new_gp, gru) if gru is not None else new_gp
            jax.block_until_ready(new_gp)
            if not bool(jnp.all(jnp.isfinite(new_gp.alpha))):
                # the float32 Cholesky can fail (condition number ~1e7): seen once, run (b) seed 1 it 11
                print(f"it {it}: policy GP fit not finite, policy not updated", flush=True)
                policy_kept = True
        else:  # safeguard: a non-finite backward pass never becomes the population's policy
            print(f"it {it}: non-finite best shares, policy not updated", flush=True)
            policy_kept = True
        if policy_kept:
            new_policy, nll = policy, jnp.nan
            new_gp = policy.gp if isinstance(policy, gpol.RecurrentPolicyGP) else policy
        step_time = time.perf_counter() - t0
        log["train_time"] += step_time

        # --- evaluation of the population's policy (before the update; not timed) ---
        row = dict(iteration=it, step_time=step_time, train_time=log["train_time"], policy_gp_nll=float(nll),
                   policy_kept=policy_kept)
        if law_of_motion:
            row["law_of_motion_nll"] = lom_nll
        if it % args.eval_frequency == 0 or it == args.num_iterations - 1:
            row.update(evaluate(policy, gps, lom))
            w_st = np.asarray(stations.mu[: T - 1]) / np.asarray(stations.mu[: T - 1]).sum(-1, keepdims=True)
            if targets.shape == pop_shares[: T - 1].shape:
                row["share_change"] = float((w_st * np.abs(np.asarray(targets) - np.asarray(pop_shares[: T - 1]))).sum(-1).mean())
            else:  # support points are not the stations: the change of the policy GP's shares at the stations
                new_shares = policy_shares(env, new_gp, jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), jax.tree.map(lambda x: x[: T - 1], stations)))
                row["share_change"] = float((w_st * np.abs(np.asarray(new_shares).reshape(pop_shares[: T - 1].shape) - np.asarray(pop_shares[: T - 1]))).sum(-1).mean())
            if isinstance(policy, (GPState, gpol.RecurrentPolicyGP)):
                if mfax_exploitability is None:
                    mfax_exploitability = make_exploitability(
                        env=env, agent=agent, state_type="states", gamma=gamma,
                        num_envs=args.eval_envs, max_steps_in_episode=T,
                    )
                res = mfax_exploitability(eval_key, policy)
                row["mfax_exploitability"] = float(res.exploitability.exploitability)
            print(
                f"it {it:3d}  train {log['train_time']:6.1f}s  "
                f"gain honest {row['gain_honest_pct']:.3f}%  clairvoyant {row['gain_clairvoyant_pct']:.3f}%  "
                + (f"clairvoyant grid {row['gain_clairvoyant_grid_pct']:.3f}%  " if grid_actions else "")
                + f"MFAX expl {row.get('mfax_exploitability', float('nan')):.4f}  "
                f"K100 {row['K_mean_at']['100']:.2f}  change {row['share_change']:.4f}"
                + (f"  LoM R2 {row['law_of_motion']['r2_of_delta_K']:.6f} max err {row['law_of_motion']['max_rel_err_pct']:.3f}%"
                   if "law_of_motion" in row else ""),
                flush=True,
            )
        log["iterations"].append(row)

        policy = new_policy
        if isinstance(new_gp, GPState):
            hyp = new_gp.hyp

    # --- save the final policy (the last update; evaluate it with experiments/gpdp/step4_evaluate.py) ---
    if args.save:
        with open(os.path.join(out_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=2)
        with open(os.path.join(out_dir, "gpdp_agent.pkl"), "wb") as f:
            agent_cls = gpol.GPDPRecurrentAgent if gru is not None else gpol.GPDPAgent
            pickle.dump(agent_cls(env, jax.device_get(policy)), f)
        print(f"Saved to {out_dir}")
    return policy, log
