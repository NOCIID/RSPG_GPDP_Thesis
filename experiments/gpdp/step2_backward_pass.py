"""
GPDP build step 2: the backward pass with V-GPs and 5 shock points, against a fixed population.

1. Stations: N economies simulated under a trained policy (default gpfix_s0).
2. Backward pass (mfax/algos/hsm/utils/gpdp_backward.py): exact best values and continuous best shares
   at the stations, period by period, with tomorrow's shock averaged over 5 points and tomorrow's
   values from the next period's GP. One GP per period.
3. Honest test on M fresh economies: a household follows the shares implied by the GPs (they use only
   information available at each period) and its value is measured along the realized paths. It is
   compared with:
   - the population policy's own value;
   - the clairvoyant best responses (grid: MFAX's br(); continuous: step 1), which know the shocks.
   Since the GP-guided household is a valid, non-clairvoyant deviation, the true non-clairvoyant best
   response lies between it and the continuous clairvoyant one.
4. GP accuracy: each period's GP prediction on the fresh economies against the realized value of the
   GP-guided household from that period on (one realized future per economy, so noisy per economy;
   the mean over economies estimates the bias).

Repeated for several numbers of stations N (nested: the first N of one simulated batch).

    python experiments/gpdp/step2_backward_pass.py
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mfax.algos.hsm.best_response import br  # noqa: E402
from mfax.algos.hsm.evaluate import evaluate_given_sequence  # noqa: E402
from mfax.algos.hsm.sequence import mf_sequence  # noqa: E402
from mfax.algos.hsm.utils import gpdp_backward as gb  # noqa: E402
from mfax.algos.hsm.utils import gpdp_continuous as gc  # noqa: E402
from mfax.algos.hsm.utils.gp_critic import aggregate_features, gp_predict  # noqa: E402
from mfax.envs import make_env  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/endogenous/hsm_rspg_gp/gpfix_s0")
    p.add_argument("--stations", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--eval-envs", type=int, default=32)
    p.add_argument("--station-seed", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=999)
    p.add_argument("--n-nodes", type=int, default=5)
    p.add_argument("--out", default="step2_backward_pass.json")
    cli = p.parse_args()

    run_dir = Path(cli.run_dir)
    run_args = json.loads((run_dir / "args.json").read_text())
    with open(run_dir / "mf_agent_wrapper.pkl", "rb") as f:
        wrapper = pickle.load(f)
    env = make_env("pushforward/endogenous", common_noise=True)
    gamma, sigma = run_args["discount_factor"], env.params.sigma
    T = env.params.max_steps_in_episode
    st = run_args["state_type"]
    ce = lambda v_hi, v_lo: (v_hi / v_lo) ** (1 / (1 - sigma)) - 1  # noqa: E731

    # --- economies: stations (one batch, nested subsets) and fresh evaluation paths ---
    stations = mf_sequence(jax.random.PRNGKey(cli.station_seed), env, wrapper, st, max(cli.stations), T)
    evals = mf_sequence(jax.random.PRNGKey(cli.eval_seed), env, wrapper, st, cli.eval_envs, T)
    mu0 = np.asarray(evals.aggregate_s.mu[0])  # (M, n_states), uniform

    def weighted(v):  # as in MFAX's exploitability: mu_0-weighted, averaged over economies
        return float((np.asarray(v) * mu0).sum(-1).mean())

    # --- references on the evaluation paths ---
    v_pol, _ = evaluate_given_sequence(env, evals, gamma=gamma)
    v_grid_clair, _, _, _ = br(env, evals, gamma)
    v_cont_clair, s_cont_clair = jax.jit(lambda a: gc.path_best_response(env, a, gamma))(evals.aggregate_s)
    ref = dict(
        policy=weighted(v_pol),
        grid_clairvoyant_br=weighted(v_grid_clair),
        continuous_clairvoyant_br=weighted(v_cont_clair),
    )
    out = dict(run_dir=str(run_dir), eval_envs=cli.eval_envs, n_nodes=cli.n_nodes, references=ref, by_stations={})
    print("references (mu_0-weighted values):", ref, flush=True)

    t_check = (1, 32, 64, 96, 120)
    for N in cli.stations:
        sub = jax.tree.map(lambda x: x[:, :N], stations.aggregate_s)
        t0 = time.perf_counter()
        gps, shares_st, values_st, nll = gb.backward_pass(env, sub, gamma, n_nodes=cli.n_nodes)
        jax.block_until_ready(values_st)
        t_backward = time.perf_counter() - t0

        # --- the GP-guided household on fresh economies ---
        t0 = time.perf_counter()
        s_greedy = gb.greedy_shares(env, evals.aggregate_s, gps, gamma, n_nodes=cli.n_nodes)
        V_greedy = gc.path_evaluate(env, evals.aggregate_s, s_greedy, gamma)  # (T, M, n_states)
        jax.block_until_ready(V_greedy)
        t_greedy = time.perf_counter() - t0

        pol, greedy = ref["policy"], weighted(V_greedy[0])
        predicted = float((np.asarray(values_st[0, 0]) * mu0[0]).sum())  # identical at all stations
        res = dict(
            backward_pass_time_s=t_backward,
            greedy_eval_time_s=t_greedy,
            values=dict(gp_guided_realized=greedy, gpdp_predicted_at_t0=predicted),
            gain_from_deviating_pct=dict(
                grid_clairvoyant_mfax=100 * ce(ref["grid_clairvoyant_br"], pol),
                continuous_clairvoyant=100 * ce(ref["continuous_clairvoyant_br"], pol),
                gp_guided_non_clairvoyant=100 * ce(greedy, pol),
            ),
            hindsight_upper_bound_pct=100 * ce(ref["continuous_clairvoyant_br"], greedy),
            gp_bias_at_t0_pct=100 * ce(predicted, greedy),
        )

        # --- GP accuracy by period on the fresh economies ---
        acc = {}
        for t in t_check:
            pred = np.asarray(gp_predict(gps[t], aggregate_features(env, jax.tree.map(lambda x: x[t], evals.aggregate_s))))
            real = np.asarray(V_greedy[t])
            mu_t = np.asarray(evals.aggregate_s.mu[t])
            w = mu_t / mu_t.sum(-1, keepdims=True)
            err = (w * ce(pred, real)).sum(-1)  # per economy, population-weighted
            acc[t] = dict(
                signed_mean_pct=100 * float(err.mean()),
                abs_mean_pct=100 * float(np.abs(err).mean()),
                se_of_mean_pct=100 * float(err.std(ddof=1) / np.sqrt(len(err))),
                nll=nll[t],
            )
        res["gp_accuracy_by_period"] = acc

        # --- shares where households are (period 64) ---
        t = 64
        mu_t = np.asarray(evals.aggregate_s.mu[t])
        w = mu_t / mu_t.sum(-1, keepdims=True)
        pol_share = np.asarray((evals.prob_a[t] * env.params.discrete_actions).sum(-1))
        res["mean_share_period_64"] = dict(
            policy=float((w * pol_share).sum(-1).mean()),
            gp_guided=float((w * np.asarray(s_greedy[t])).sum(-1).mean()),
            continuous_clairvoyant=float((w * np.asarray(s_cont_clair[t])).sum(-1).mean()),
        )
        out["by_stations"][N] = res
        print(f"N = {N}:", json.dumps(res, indent=2), flush=True)

    path = Path(__file__).parent / "results" / cli.out
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
