"""
Check 1: does the truncated gradient point where the full-rollout gradient points?

At fixed policy parameters, estimate four gradients by averaging many independent batches:
    full      : original objective (every economy resets, 128-period rollouts, no critic)
    gp        : staggered 25-period windows + GP tail (the method)
    zero_tail : staggered 25-period windows, tail = 0 (truncation without a critic)
    gp_reset  : 25-period windows from the reset only + GP tail (no staggering)
    gp_time   : staggered 32-period windows covering the whole episode + time-aware GP tail
and report each one's cosine with, and norm relative to, the full gradient. Split-half cosines
(two independent halves of the same estimator) show how much of the gap is Monte Carlo noise.

    python experiments/gp_critic/check_gradient.py --run-dir runs/endogenous/hsm_rspg_gp/gp_s0
"""

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.flatten_util
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from mfax.algos.hsm.utils.gp_train import (  # noqa: E402
    build_policy,
    init_actor,
    initial_econ,
    make_loss,
    phase_layout,
    warmstart_targets,
)
from mfax.envs import make_env  # noqa: E402


def _flat(tree):
    return np.asarray(jax.flatten_util.ravel_pytree(tree)[0], dtype=np.float64)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def estimate(args, env, policy_fn, params, gp, n_batches, key):
    """Average gradient over n_batches independent batches; also return the two half-averages."""
    layout = phase_layout(args.phase_counts, args.horizon, args.discount_factor)
    loss_fn = make_loss(args, env, policy_fn, layout)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    key, k_econ = jax.random.split(key)
    econ = initial_econ(args, env, policy_fn, layout, k_econ)
    # --- fill the pipeline under these parameters ---
    for _ in range(layout.n_phases - 1):
        key, k = jax.random.split(key)
        (_, (econ, _, _, _)), _ = grad_fn(params, gp, econ, k)
    grads = []
    for _ in range(n_batches):
        key, k = jax.random.split(key)
        (_, (econ, _, _, _)), g = grad_fn(params, gp, econ, k)
        grads.append(_flat(g))
    grads = np.stack(grads)
    half = n_batches // 2
    return grads.mean(0), grads[:half].mean(0), grads[half:].mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="a GP run (for the trained policy)")
    p.add_argument("--full-batches", type=int, default=16)
    p.add_argument("--window-batches", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--variants", nargs="+", default=["full", "gp", "zero_tail", "gp_reset", "gp_time"])
    p.add_argument("--out", default=None)
    p.add_argument(
        "--use-run-critic", action="store_true",
        help="at the trained policy, use the run's own trained critic instead of a fresh warm-start fit",
    )
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
    init_ts, _ = init_actor(base, env, net, individual_states, recurrent)

    variants = dict(
        full=dict(use_gp=False, horizon=128, phase_counts=(8,)),
        gp=dict(use_gp=True, horizon=25, phase_counts=(4, 2, 1, 1)),
        zero_tail=dict(use_gp=False, horizon=25, phase_counts=(4, 2, 1, 1)),
        gp_reset=dict(use_gp=True, horizon=25, phase_counts=(8,)),
        gp_time=dict(use_gp=True, horizon=32, phase_counts=(4, 2, 1, 1), time_input=True),
    )
    variants = {k: dict(time_input=False, **v) if "time_input" not in v else v
                for k, v in variants.items() if k in cli.variants}
    out = {}
    key = jax.random.PRNGKey(cli.seed)
    for label, params in [("initial_policy", init_ts.params), ("trained_policy", trained_params)]:
        # --- a critic for *these* parameters, fitted as in training (warm start), one per kind ---
        critics = {}
        for time_input in sorted({v["time_input"] for v in variants.values()}):
            args = replace(base, time_input=time_input)
            key, k_ws, k_sub = jax.random.split(key, 3)
            X, Y = warmstart_targets(args, env, policy_fn, params, k_ws)
            Xs, Ys = subsample(k_sub, X, Y, base.n_gp)
            gp, _ = fit_gp(
                Xs, Ys, init_hyper(X.shape[-1], n_base_inputs(time_input)),
                base.gp_steps_first, base.gp_lr,
            )
            critics[time_input] = (gp, empty_gp(base.n_gp, X.shape[-1], env.n_states))
        if cli.use_run_critic and label == "trained_policy":
            with open(run_dir / "gp_state.pkl", "rb") as f:
                run_gp = pickle.load(f)
            kind = run_args.get("time_input", False)
            critics[kind] = (run_gp, critics[kind][1])
            print(f"using the run's trained critic for time_input={kind}", flush=True)

        res = {}
        for name, overrides in variants.items():
            args = replace(base, **overrides)
            n = cli.full_batches if name == "full" else cli.window_batches
            key, k = jax.random.split(key)
            gp, no_gp = critics[args.time_input]
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
                # --- Monte Carlo noise of this estimator: distance between its two halves ---
                split_half_rel_diff=float(np.linalg.norm(r["h1"] - r["h2"]) / nf),
                split_half_cos=_cos(r["h1"], r["h2"]),
            )
            for name, r in res.items()
        }
        # --- expected rel_err_vs_full from Monte Carlo noise alone (a mean is off by ~half the
        #     distance between its two halves; noises of the two estimators add in quadrature) ---
        for name in summary:
            summary[name]["noise_floor_rel_err"] = float(
                np.hypot(summary["full"]["split_half_rel_diff"], summary[name]["split_half_rel_diff"]) / 2
            )
        out[label] = summary
        print(label, json.dumps(summary, indent=2), flush=True)

    tag = "rspg" if recurrent else "spg"
    path = Path(__file__).parent / "results" / (cli.out or f"gradient_{tag}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
