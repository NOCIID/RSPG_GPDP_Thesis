"""
Check 4: compare trained runs.

- Exploitability and policy return on a fresh, larger evaluation set (a key never used in training).
- Timing: steady time per step, total training time, time to reach exploitability thresholds.
- Policies: expected consumption share E[a | x] of each policy against a reference policy, on a
  common path (the reference policy's own episodes; each policy runs its GRU on the same
  observations), over all 128 periods. Gaps are relative (= relative consumption gaps, since
  c = a * cash-on-hand), weighted by the population distribution, with maxima over visited states.
  The seed-to-seed gap of the reference method gives the noise baseline.

    python experiments/gp_critic/compare_runs.py --ref runs/endogenous/hsm_rspg_gp/base_s0 \
        --runs runs/endogenous/hsm_rspg_gp/base_s1 runs/endogenous/hsm_rspg_gp/gp_s0 ...
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mfax.algos.hsm.exploitability import make_exploitability  # noqa: E402
from mfax.algos.hsm.sequence import mf_sequence  # noqa: E402
from mfax.algos.hsm.utils.gp_rollouts import MFPolicyFn  # noqa: E402
from mfax.envs import make_env  # noqa: E402


def load(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "mf_agent_wrapper.pkl", "rb") as f:
        wrapper = pickle.load(f)
    return dict(
        name=f"{run_dir.parent.name}/{run_dir.name}",
        args=json.loads((run_dir / "args.json").read_text()),
        log=json.loads((run_dir / "log.json").read_text()),
        wrapper=wrapper,
    )


def mean_actions_on_path(env, policy_fn, params, obs_seq):
    """E[a | x] along a fixed observation sequence (T, E, obs_dim); the policy runs its own GRU."""
    E = obs_seq.shape[1]
    done = jnp.zeros((E,), dtype=jnp.bool_)

    def _step(h, obs):
        prob_a, h_next = policy_fn(params, h, obs, done)
        return h_next, policy_fn.mean_action(prob_a)

    _, A = jax.lax.scan(_step, policy_fn.init_hidden(E), obs_seq)
    return np.asarray(A)  # (T, E, n_states)


def step_time_median(timing, refit_every):
    """Median time per step, leaving out the first chunk of each length (compilation)."""
    steps = timing.get("chunk_steps", [refit_every] * len(timing["chunks"]))
    per_step, seen = [], set()
    for t, n in zip(timing["chunks"], steps):
        if n in seen:
            per_step.append(t / n)
        seen.add(n)
    return float(np.median(per_step))


def time_to(curve, threshold):
    for point in curve:
        if point["exploitability"] <= threshold:
            return point["train_time"]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="reference run (original method, seed 0)")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--eval-envs", type=int, default=64)
    p.add_argument("--eval-seed", type=int, default=777)
    p.add_argument("--path-envs", type=int, default=16)
    p.add_argument("--out", default="compare.json")
    cli = p.parse_args()

    env = make_env("pushforward/endogenous", common_noise=True)
    ref = load(cli.ref)
    runs = [ref] + [load(r) for r in cli.runs]
    state_type = ref["args"]["state_type"]
    gamma = ref["args"]["discount_factor"]
    T = env.params.max_steps_in_episode
    out = dict(eval_envs=cli.eval_envs, eval_seed=cli.eval_seed, runs={})

    # --- common path: the reference policy's episodes ---
    path = mf_sequence(
        jax.random.PRNGKey(cli.eval_seed + 1), env, ref["wrapper"], state_type, cli.path_envs, T
    )
    obs_seq = path.aggregate_obs  # (T, E, obs_dim)
    mu = np.asarray(path.aggregate_s.mu)  # (T, E, n_states)
    w = mu / mu.sum(-1, keepdims=True) / T
    visited = mu > 1e-4
    ref_fn = MFPolicyFn.from_wrapper(env, ref["wrapper"], state_type)
    a_ref = mean_actions_on_path(env, ref_fn, ref["wrapper"].mf_params, obs_seq)
    K_ref = None

    for run in runs:
        wrapper = run["wrapper"]
        expl_fn = make_exploitability(
            env=env,
            agent=wrapper,
            state_type=state_type,
            gamma=gamma,
            num_envs=cli.eval_envs,
            max_steps_in_episode=T,
        )
        res = expl_fn(jax.random.PRNGKey(cli.eval_seed), wrapper.mf_params)
        policy_fn = MFPolicyFn.from_wrapper(env, wrapper, state_type)
        a = mean_actions_on_path(env, policy_fn, wrapper.mf_params, obs_seq)
        rel = np.abs(a / a_ref - 1)

        # --- each policy's own aggregate path on the same shocks ---
        own = mf_sequence(
            jax.random.PRNGKey(cli.eval_seed + 1), env, wrapper, state_type, cli.path_envs, T
        )
        mu_own = np.asarray(own.aggregate_s.mu).reshape(T, cli.path_envs, *env.params.num_states)
        K_own = (mu_own.sum(-1) * np.asarray(env.params.discrete_states[0])).sum(-1)
        r_own = np.asarray(own.aggregate_s.interest_rate)
        if K_ref is None:
            K_ref, r_ref = K_own, r_own

        timing = run["log"]["timing"]
        curve = run["log"]["curve"]
        out["runs"][run["name"]] = dict(
            exploitability=float(res.exploitability.exploitability),
            policy_return=float(res.exploitability.mean_policy_return),
            br_return=float(res.exploitability.mean_br_return),
            final_training_eval_exploitability=curve[-1]["exploitability"],
            total_train_time_s=timing["total_train_time"],
            step_time_steady_s=timing["step_time_steady"],
            # --- median over chunks: robust to chunks slowed by sleep or power state ---
            step_time_median_s=step_time_median(timing, run["args"]["refit_every"]),
            refit_time_median_s=(
                float(np.median(timing["refits"])) if timing["refits"] else None
            ),
            first_chunk_incl_compile_s=timing["first_chunk_incl_compile"],
            refit_time_steady_s=timing.get("refit_time_steady"),
            warmstart_s=timing.get("warmstart"),
            fill_s=timing.get("fill"),
            mean_eval_time_s=float(np.mean(timing["evals"])),
            curve=curve,
            policy_gap_vs_ref=dict(
                pop_weighted_mean_rel=float((w * rel).sum() / cli.path_envs),
                visited_max_rel=float(rel[visited].max()),
                visited_p99_rel=float(np.quantile(rel[visited], 0.99)),
            ),
            aggregate_gap_vs_ref=dict(
                K_max_rel=float(np.abs(K_own / K_ref - 1).max()),
                K_mean_rel=float(np.abs(K_own / K_ref - 1).mean()),
                r_max_abs=float(np.abs(r_own - r_ref).max()),
                r_mean_abs=float(np.abs(r_own - r_ref).mean()),
            ),
            # --- average aggregate path over the common shocks (periods 0, 25, 50, 100, 127) ---
            aggregate_path=dict(
                periods=[0, 25, 50, 100, T - 1],
                K_mean=[float(K_own[t].mean()) for t in (0, 25, 50, 100, T - 1)],
                r_mean=[float(r_own[t].mean()) for t in (0, 25, 50, 100, T - 1)],
            ),
            mean_consumption_share=float(
                (mu / mu.sum(-1, keepdims=True) * a).sum(-1).mean()
            ),
        )
        print(run["name"], json.dumps({k: v for k, v in out["runs"][run["name"]].items() if k != "curve"}, indent=2), flush=True)

    # --- time to reach thresholds relative to the reference's final training-eval exploitability ---
    ref_final = out["runs"][ref["name"]]["final_training_eval_exploitability"]
    for name, r in out["runs"].items():
        r["time_to_threshold_s"] = {
            f"{f:.2f}x_ref_final": time_to(r["curve"], f * ref_final) for f in (2.0, 1.5, 1.2, 1.1)
        }

    path_out = Path(__file__).parent / "results" / cli.out
    path_out.write_text(json.dumps(out, indent=2))
    print(f"Saved {path_out}")


if __name__ == "__main__":
    main()
