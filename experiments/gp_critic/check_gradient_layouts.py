"""
Check 1 for phase layouts: the windowed gradient against the full 128-period gradient, for the
default layout (4, 2, 1, 1) and for (6, 2), at the initial policy (fresh warm-start critic) and at a
run's trained policy (that run's own critic, as in check_gradient.py --use-run-critic).

(6, 2) gives no gradient to periods 64-127, so even with a perfect critic its expectation misses
their contribution; at the initial policy the fresh critic is accurate, which isolates that part.

    python experiments/gp_critic/check_gradient_layouts.py \
        --run-dir runs/endogenous/hsm_rspg_gp/gp62_s0 --out gradient_layouts_rspg_gp62_s0.json
"""

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_gradient import _cos, estimate  # noqa: E402
from mfax.algos.hsm.algos.rspg_gp import args as RSPGGPArgs  # noqa: E402
from mfax.algos.hsm.algos.spg_gp import args as SPGGPArgs  # noqa: E402
from mfax.algos.hsm.utils.gp_critic import (  # noqa: E402
    empty_gp,
    fit_gp,
    init_hyper,
    n_base_inputs,
    subsample,
)
from mfax.algos.hsm.utils.gp_rollouts import MFPolicyFn  # noqa: E402
from mfax.algos.hsm.utils.gp_train import build_policy, init_actor, warmstart_targets  # noqa: E402
from mfax.envs import make_env  # noqa: E402

VARIANTS = dict(
    full=dict(use_gp=False, horizon=128, phase_counts=(8,), time_input=True),
    layout_4211=dict(use_gp=True, horizon=32, phase_counts=(4, 2, 1, 1), time_input=True),
    layout_62=dict(use_gp=True, horizon=32, phase_counts=(6, 2), time_input=True),
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="a time-aware GP run (trained policy + its critic)")
    p.add_argument("--full-batches", type=int, default=16)
    p.add_argument("--window-batches", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-initial", action="store_true")
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    run_dir = Path(cli.run_dir)
    run_args = json.loads((run_dir / "args.json").read_text())
    recurrent = run_args["algo"].startswith("hsm_rspg")
    base = (RSPGGPArgs if recurrent else SPGGPArgs)()
    base = replace(
        base,
        **{k: (tuple(v) if isinstance(v, list) else v) for k, v in run_args.items()
           if k in base.__dataclass_fields__},
    )
    env = make_env("pushforward/endogenous", common_noise=True)
    individual_states = env.params.states
    net = build_policy(base, env, recurrent)
    policy_fn = MFPolicyFn(env, net, individual_states, base.normalize_obs, base.normalize_states, recurrent)
    with open(run_dir / "mf_agent_wrapper.pkl", "rb") as f:
        trained_params = pickle.load(f).mf_params
    with open(run_dir / "gp_state.pkl", "rb") as f:
        run_gp = pickle.load(f)
    init_ts, _ = init_actor(base, env, net, individual_states, recurrent)

    policies = [("trained_policy", trained_params)]
    if not cli.skip_initial:
        policies = [("initial_policy", init_ts.params)] + policies
    out = dict(run_dir=str(run_dir), run_phase_counts=run_args["phase_counts"])
    key = jax.random.PRNGKey(cli.seed)
    for label, params in policies:
        args_t = replace(base, time_input=True)
        if label == "trained_policy":
            gp = run_gp
            critic = "the run's own critic"
        else:
            key, k_ws, k_sub = jax.random.split(key, 3)
            X, Y = warmstart_targets(args_t, env, policy_fn, params, k_ws)
            Xs, Ys = subsample(k_sub, X, Y, base.n_gp)
            gp, _ = fit_gp(Xs, Ys, init_hyper(X.shape[-1], n_base_inputs(True)), base.gp_steps_first, base.gp_lr)
            critic = "fresh warm-start fit"
        no_gp = empty_gp(base.n_gp, gp.X.shape[-1], env.n_states)

        res = {}
        for name, overrides in VARIANTS.items():
            args = replace(base, **overrides)
            n = cli.full_batches if name == "full" else cli.window_batches
            key, k = jax.random.split(key)
            mean, h1, h2 = estimate(args, env, policy_fn, params, gp if args.use_gp else no_gp, n, k)
            res[name] = dict(mean=mean, h1=h1, h2=h2, n_batches=n)
            print(label, name, "done", flush=True)
        full = res["full"]["mean"]
        nf = np.linalg.norm(full)
        summary = {
            name: dict(
                n_batches=r["n_batches"],
                rel_err_vs_full=float(np.linalg.norm(r["mean"] - full) / nf),
                cos_with_full=_cos(r["mean"], full),
                norm_ratio_to_full=float(np.linalg.norm(r["mean"]) / nf),
                split_half_rel_diff=float(np.linalg.norm(r["h1"] - r["h2"]) / nf),
            )
            for name, r in res.items()
        }
        for name in summary:
            summary[name]["noise_floor_rel_err"] = float(
                np.hypot(summary["full"]["split_half_rel_diff"], summary[name]["split_half_rel_diff"]) / 2
            )
        out[label] = dict(critic=critic, **summary)
        print(label, json.dumps(out[label], indent=2), flush=True)

    path = Path(__file__).parent / "results" / cli.out
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
