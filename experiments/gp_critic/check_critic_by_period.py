"""
Check 2 by test period: the critic's error against brute-force Monte Carlo at each window end
separately (periods H, 2H, 3H), for several runs. Same test states, Monte Carlo reference and
scores as check 2a in check_critic.py (same key sequence, so a run's pooled score matches
check_critic.py's), but each period is scored on its own, with absolute as well as signed errors.

Used to compare phase layouts whose windows end at different periods, e.g. (4, 2, 1, 1) uses the
critic at periods 32, 64 and 96, while (6, 2) uses it at 32 and 64 only.

    python experiments/gp_critic/check_critic_by_period.py \
        --run-dirs runs/endogenous/hsm_rspg_gp/gpfix_s0 runs/endogenous/hsm_rspg_gp/gp62_s0 \
        --out critic_by_period_gp62.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_critic import inputs, mc_values, score, test_states  # noqa: E402
from mfax.algos.hsm.utils.gp_critic import gp_predict  # noqa: E402
from mfax.algos.hsm.utils.gp_rollouts import MFPolicyFn  # noqa: E402
from mfax.envs import make_env  # noqa: E402


def check_run(env, run_dir, cli):
    run_args = json.loads((run_dir / "args.json").read_text())
    with open(run_dir / "mf_agent_wrapper.pkl", "rb") as f:
        wrapper = pickle.load(f)
    with open(run_dir / "gp_state.pkl", "rb") as f:
        gp = pickle.load(f)
    T = env.params.max_steps_in_episode
    policy_fn = MFPolicyFn.from_wrapper(env, wrapper, run_args["state_type"])
    params = wrapper.mf_params
    sigma = env.params.sigma
    assert run_args.get("time_input", False), "finite-horizon reference: time-aware critics only"
    H = run_args["horizon"]
    times = (H, 2 * H, 3 * H)
    # --- window ends with a GP tail: H, 2H, ..., n_phases * H, except the episode's true end ---
    used = tuple(H * k for k in range(1, len(run_args["phase_counts"]) + 1) if H * k < T)

    # --- same key sequence as check_critic.py ---
    key = jax.random.PRNGKey(cli.seed)
    key, k_test, k_mc = jax.random.split(key, 3)
    state = test_states(env, policy_fn, params, k_test, times, cli.per_time)
    s = state[0]
    mu = np.asarray(s.mu)
    v_mc, se_mc = mc_values(
        env, policy_fn, params, run_args["discount_factor"], state, k_mc, cli.mc, T - min(times),
        cli.chunk, finite=True,
    )
    x_test = inputs(state, True, T)
    v_gp = np.asarray(gp_predict(gp, jnp.asarray(x_test[:, : gp.X.shape[1]])))

    def rows(ts):
        idx = np.concatenate([np.arange(times.index(t) * cli.per_time, (times.index(t) + 1) * cli.per_time) for t in ts])
        return score(v_gp[idx], v_mc[idx], se_mc[idx], mu[idx], sigma)

    return dict(
        phase_counts=run_args["phase_counts"],
        critic_used_at=used,
        pooled_all_test_periods=rows(times),
        pooled_used_periods=rows(used),
        by_period={int(t): rows((t,)) for t in times},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--per-time", type=int, default=8)
    p.add_argument("--mc", type=int, default=32)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    env = make_env("pushforward/endogenous", common_noise=True)
    out = dict(per_time=cli.per_time, mc=cli.mc, seed=cli.seed, runs={})
    for d in cli.run_dirs:
        run_dir = Path(d)
        name = f"{run_dir.parent.name}/{run_dir.name}"
        out["runs"][name] = check_run(env, run_dir, cli)
        r = out["runs"][name]
        print(
            name,
            "pooled(all)", round(100 * r["pooled_all_test_periods"]["ce_err_pop_weighted_mean"], 2),
            "pooled(used)", round(100 * r["pooled_used_periods"]["ce_err_pop_weighted_mean"], 2),
            {t: (round(100 * v["ce_err_pop_weighted_mean"], 2), round(100 * v["ce_err_signed_pop_weighted_mean"], 2))
             for t, v in r["by_period"].items()},
            flush=True,
        )
    path = Path(__file__).parent / "results" / cli.out
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
