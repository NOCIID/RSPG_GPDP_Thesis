"""
GPDP build step 3: the policy GP.

1. Backward pass against a fixed population (step 2), on N stations: best shares at the stations.
2. Fit one policy GP on them (mfax/algos/hsm/utils/gpdp_policy.py): inputs (z, K, T - t), one output
   per individual state, fitted on log(a / (1 - a)).
3. On M fresh economies, compare a household following the policy GP with the step-2 household,
   which computes its shares exactly by lookahead with the V-GPs:
   - shares: the policy GP's against the lookahead's, by period;
   - value: both realized along the fresh paths, and against the population policy;
   - cost: predicting shares with the policy GP against computing them by lookahead.
The population is still fixed (step 4 lets it adopt the policy GP).

    python experiments/gpdp/step3_policy_gp.py
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mfax.algos.hsm.evaluate import evaluate_given_sequence  # noqa: E402
from mfax.algos.hsm.sequence import mf_sequence  # noqa: E402
from mfax.algos.hsm.utils import gpdp_backward as gb  # noqa: E402
from mfax.algos.hsm.utils import gpdp_continuous as gc  # noqa: E402
from mfax.algos.hsm.utils import gpdp_policy as gpol  # noqa: E402
from mfax.envs import make_env  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/endogenous/hsm_rspg_gp/gpfix_s0")
    p.add_argument("--stations", type=int, default=8)
    p.add_argument("--eval-envs", type=int, default=32)
    p.add_argument("--station-seed", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=999)
    p.add_argument("--gp-steps", type=int, default=300)
    p.add_argument("--out", default="step3_policy_gp.json")
    cli = p.parse_args()

    run_dir = Path(cli.run_dir)
    run_args = json.loads((run_dir / "args.json").read_text())
    with open(run_dir / "mf_agent_wrapper.pkl", "rb") as f:
        wrapper = pickle.load(f)
    env = make_env("pushforward/endogenous", common_noise=True)
    gamma, sigma = run_args["discount_factor"], env.params.sigma
    T, st = env.params.max_steps_in_episode, run_args["state_type"]
    ce = lambda v_hi, v_lo: (v_hi / v_lo) ** (1 / (1 - sigma)) - 1  # noqa: E731

    stations = mf_sequence(jax.random.PRNGKey(cli.station_seed), env, wrapper, st, cli.stations, T)
    evals = mf_sequence(jax.random.PRNGKey(cli.eval_seed), env, wrapper, st, cli.eval_envs, T)
    mu0 = np.asarray(evals.aggregate_s.mu[0])
    weighted = lambda v: float((np.asarray(v) * mu0).sum(-1).mean())  # noqa: E731

    # --- 1. backward pass (step 2) ---
    t0 = time.perf_counter()
    gps, shares_st, _, _ = gb.backward_pass(env, stations.aggregate_s, gamma)
    jax.block_until_ready(shares_st)
    out = dict(stations=cli.stations, eval_envs=cli.eval_envs, backward_pass_time_s=time.perf_counter() - t0)

    # --- 2. policy GP ---
    X, Y = gpol.training_data(env, stations.aggregate_s, shares_st)
    t0 = time.perf_counter()
    pgp, nll = gpol.fit_policy_gp(X, Y, n_steps=cli.gp_steps)
    jax.block_until_ready(pgp)
    fitted = np.asarray(jax.nn.sigmoid(gb.gp_predict(pgp, X)))
    target = np.asarray(jax.nn.sigmoid(Y))
    out["policy_gp"] = dict(
        n_train=int(X.shape[0]),
        fit_time_s=time.perf_counter() - t0,
        nll=float(nll),
        length_scales_z_K_timeleft=np.exp(np.asarray(pgp.hyp.log_ell)).round(3).tolist(),
        signal_std=float(np.exp(pgp.hyp.log_sf)),
        noise_std=float(np.exp(pgp.hyp.log_sn)),
        train_abs_share_err_mean=float(np.abs(fitted - target).mean()),
    )
    print("policy GP:", out["policy_gp"], flush=True)

    # --- 3. fresh economies: lookahead (step 2) vs policy GP ---
    t0 = time.perf_counter()
    s_look = gb.greedy_shares(env, evals.aggregate_s, gps, gamma)
    jax.block_until_ready(s_look)
    t_look = time.perf_counter() - t0
    path_shares = jax.jit(lambda a: gpol.path_shares(env, pgp, a))
    jax.block_until_ready(path_shares(evals.aggregate_s))  # compile
    t0 = time.perf_counter()
    s_pgp = jax.block_until_ready(path_shares(evals.aggregate_s))
    t_pgp = time.perf_counter() - t0
    out["share_time_s"] = dict(lookahead=t_look, policy_gp=t_pgp)

    V_look = gc.path_evaluate(env, evals.aggregate_s, s_look, gamma)[0]
    V_pgp = gc.path_evaluate(env, evals.aggregate_s, s_pgp, gamma)[0]
    v_pol, _ = evaluate_given_sequence(env, evals, gamma=gamma)
    v_clair, _ = jax.jit(lambda a: gc.path_best_response(env, a, gamma))(evals.aggregate_s)
    pol, look, pg, clair = map(weighted, (v_pol, V_look, V_pgp, v_clair))
    out["values"] = dict(policy=pol, lookahead=look, policy_gp=pg, continuous_clairvoyant=clair)
    out["gain_over_population_policy_pct"] = dict(
        lookahead=100 * ce(look, pol), policy_gp=100 * ce(pg, pol), continuous_clairvoyant=100 * ce(clair, pol)
    )
    out["policy_gp_loss_vs_lookahead_pct"] = 100 * ce(look, pg)

    # --- share accuracy by period, weighted by where households are ---
    acc = {}
    s_look_np, s_pgp_np = np.asarray(s_look), np.asarray(s_pgp)
    for t in (0, 1, 32, 64, 96, 120, 126):
        mu_t = np.asarray(evals.aggregate_s.mu[t])
        w = mu_t / mu_t.sum(-1, keepdims=True)
        diff = np.abs(s_pgp_np[t] - s_look_np[t])
        rel = diff / s_look_np[t]
        visited = mu_t > 1e-4
        acc[t] = dict(
            abs_err_pop_weighted=float((w * diff).sum(-1).mean()),
            rel_err_pop_weighted=float((w * rel).sum(-1).mean()),
            rel_err_visited_max=float(rel[visited].max()),
            mean_share_lookahead=float((w * s_look_np[t]).sum(-1).mean()),
            mean_share_policy_gp=float((w * s_pgp_np[t]).sum(-1).mean()),
        )
    out["share_accuracy_by_period"] = acc

    print(json.dumps({k: v for k, v in out.items() if k != "policy_gp"}, indent=2))
    path = Path(__file__).parent / "results" / cli.out
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
