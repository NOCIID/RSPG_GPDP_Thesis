"""
Check 5: does a policy anticipate the end of the 128-period episode?

The original method optimizes the finite-horizon objective, and RSPG's GRU can act as a clock.
The GP version uses an infinite-horizon critic and trains periods 0-99 only. For each policy, on
its own episodes (same aggregate shocks for every policy), report the population-average
consumption share and capital by period. Rising consumption shares near period 127 indicate
end-of-episode dissaving.

    python experiments/gp_critic/check_horizon.py --runs runs/endogenous/hsm_rspg_gp/base_s0 ...
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mfax.algos.hsm.sequence import mf_sequence  # noqa: E402
from mfax.algos.hsm.utils.gp_rollouts import MFPolicyFn  # noqa: E402
from mfax.envs import make_env  # noqa: E402

BLOCKS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 115), (115, 123), (123, 128)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=778)
    p.add_argument("--out", default="horizon.json")
    cli = p.parse_args()

    env = make_env("pushforward/endogenous", common_noise=True)
    T = env.params.max_steps_in_episode
    out = dict(blocks=BLOCKS, runs={})
    for run_dir in map(Path, cli.runs):
        args = json.loads((run_dir / "args.json").read_text())
        with open(run_dir / "mf_agent_wrapper.pkl", "rb") as f:
            wrapper = pickle.load(f)
        seq = mf_sequence(
            jax.random.PRNGKey(cli.seed), env, wrapper, args["state_type"], cli.envs, T
        )
        policy_fn = MFPolicyFn.from_wrapper(env, wrapper, args["state_type"])
        mu = np.asarray(seq.aggregate_s.mu)  # (T, E, n_states)
        share = np.asarray(policy_fn.mean_action(seq.prob_a))  # (T, E, n_states)
        c_share = (mu * share).sum(-1).mean(-1)  # population-average consumption share per period
        K = (
            mu.reshape(T, cli.envs, *env.params.num_states).sum(-1)
            * np.asarray(env.params.discrete_states[0])
        ).sum(-1).mean(-1)
        name = f"{run_dir.parent.name}/{run_dir.name}"
        out["runs"][name] = dict(
            consumption_share=[float(c_share[a:b].mean()) for a, b in BLOCKS],
            K=[float(K[a:b].mean()) for a, b in BLOCKS],
            consumption_share_last5=c_share[-5:].round(4).tolist(),
        )
        print(name, json.dumps(out["runs"][name]), flush=True)

    path = Path(__file__).parent / "results" / cli.out
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
