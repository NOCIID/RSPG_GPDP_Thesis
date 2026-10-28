"""
Checks for the law-of-motion GP and the closed-form expectation (mfax/algos/hsm/utils/gpdp_lom.py),
against a fixed population: the final policy of a report run (single value GP, alpha = 1).

1. Law of motion: fitted on 8 stations, it predicts K_{t+1} on 64 fresh economies. Error, R^2 of
   K_{t+1} - K_t, calibration of its variance; compared with the Krusell-Smith log-linear rule
   log K' = a + b log K + c z (+ d (T - t)) fitted on the same stations.
2. Closed form vs Gauss-Hermite, K' exact: the continuation E_z'[V_{t+1}] at the stations, both ways.
3. Closed form vs Monte Carlo, K' uncertain (law-of-motion variance).
4. Whole backward pass at the stations, K' exact (report) vs law of motion + closed form.

    python experiments/gpdp/lom_checks.py --run runs/endogenous/gpdp/report_singlevgp_a1_s0
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

from mfax.algos.hsm.utils import gpdp_backward as gb  # noqa: E402
from mfax.algos.hsm.utils import gpdp_lom as glom  # noqa: E402
from mfax.algos.hsm.utils import gpdp_train as gt  # noqa: E402
from mfax.algos.hsm.utils.gp_critic import aggregate_features  # noqa: E402
from mfax.envs import make_env  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def ks_rule(X, K_next):
    """Least squares log K' = a + b log K + c z + d (T - t) on (z, K, T - t) inputs."""
    A = np.column_stack([np.ones(len(X)), np.log(X[:, 1]), X[:, 0], X[:, 2]])
    coef, *_ = np.linalg.lstsq(A, np.log(K_next), rcond=None)
    return lambda X: np.exp(np.column_stack([np.ones(len(X)), np.log(X[:, 1]), X[:, 0], X[:, 2]]) @ coef)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/endogenous/gpdp/report_singlevgp_a1_s0")
    p.add_argument("--stations", type=int, default=8)
    p.add_argument("--fresh", type=int, default=64)
    p.add_argument("--mc", type=int, default=20000)
    p.add_argument("--out", default="lom_checks.json")
    cli = p.parse_args()

    env = make_env("pushforward/endogenous", common_noise=True)
    gamma, sigma, T = 0.95, env.params.sigma, env.params.max_steps_in_episode
    with open(Path(cli.run) / "gpdp_agent.pkl", "rb") as f:
        policy = pickle.load(f).mf_params
    stations, _ = gt.forward(env, policy, jax.random.PRNGKey(4242), cli.stations)
    fresh, _ = gt.forward(env, policy, jax.random.PRNGKey(777), cli.fresh)
    out = dict(run=cli.run)

    # --- average income is constant along paths (prices at any (z, K) rely on it) ---
    L_all = np.asarray(glom.average_income(env, stations.mu))
    out["average_income_max_abs_change"] = float(np.abs(L_all - L_all[0]).max())

    # --- 1. law of motion ---
    X, y = glom.transitions(env, stations)
    t0 = time.perf_counter()
    lom, nll = glom.fit_law_of_motion(X, y, n_steps=300)
    fit_time = time.perf_counter() - t0
    Xf, yf = (np.asarray(a) for a in glom.transitions(env, fresh))
    K_next = Xf[:, 1] + yf[:, 0]
    m, v = (np.asarray(a) for a in glom.predict_next_K(lom, jnp.asarray(Xf)))
    zscore = (K_next - m) / np.sqrt(v)
    ks = ks_rule(np.asarray(X), np.asarray(X[:, 1] + y[:, 0]))(Xf)
    dK = yf[:, 0]

    def stats(pred):
        err = K_next - pred
        return dict(
            rmse=float(np.sqrt(np.mean(err**2))),
            max_rel_err_pct=float(100 * np.max(np.abs(err) / K_next)),
            r2_of_delta_K=float(1 - np.sum(err**2) / np.sum((dK - dK.mean()) ** 2)),
        )

    out["law_of_motion"] = dict(
        fit_time_s=fit_time,
        nll=nll,
        n_train=int(X.shape[0]),
        length_scales_std_units_z_K_logtau=np.exp(np.asarray(lom.hyp.log_ell)).tolist(),
        noise_std_units=float(np.exp(lom.hyp.log_sn)),
        gp=stats(m),
        krusell_smith_loglinear=stats(ks),
        no_change_K_next_eq_K=stats(Xf[:, 1]),
        mean_K=float(K_next.mean()),
        mean_std_K_next=float(np.sqrt(v).mean()),
        zscore_std=float(zscore.std()),
        coverage_95_pct=float(100 * np.mean(np.abs(zscore) < 1.96)),
    )
    print(json.dumps(out["law_of_motion"], indent=1), flush=True)

    # --- value GP of a backward pass with exact K' (the report's), used by checks 2 and 3 ---
    t0 = time.perf_counter()
    gps, best_ex, val_ex, _ = gb.backward_pass_single(env, stations, gamma)
    jax.block_until_ready(val_ex)
    time_exact = time.perf_counter() - t0
    gp = gps[1]
    eps, probs = gb.shock_nodes(5)
    eps40, probs40 = gb.shock_nodes(40)  # reference for the expectation over z'
    feats = jax.vmap(lambda a: aggregate_features(env, a))(stations)  # (T, N, 2)

    # --- 2. closed form vs Gauss-Hermite, K' exact; 3. closed form vs Monte Carlo, K' from the law ---
    rel_gh, rel_gh_nocorr, rel_mc, corr, rel_gh5 = [], [], [], [], []
    key = jax.random.PRNGKey(0)
    for t in (1, 16, 32, 64, 96, 120, 125):
        now, nxt = gb._period(stations, t), gb._period(stations, t + 1)
        for e in range(cli.stations):
            a = jax.tree.map(lambda x: x[e], now)
            an = jax.tree.map(lambda x: x[e], nxt)
            K1 = feats[t + 1, e, 1]
            w_gh = gb.continuation(env, a, an, gp, eps40, probs40, False, gamma)
            w_gh5 = gb.continuation(env, a, an, gp, eps, probs, False, gamma)
            rel_gh5.append(float(jnp.max(jnp.abs(w_gh5 / w_gh - 1))))
            w_cf = glom.continuation(env, a, K1, jnp.float32(0.0), gp, gamma)
            tl = T - (t + 1)
            y_mean, _ = glom.expected_prediction(
                gp, jnp.stack([env.params.rho * a.z, K1, tl]), jnp.stack([env.params.nu**2, 0.0, 0.0])
            )
            w_nc = gb.log_ce_to_value(y_mean, tl, gamma, sigma)
            rel_gh.append(float(jnp.max(jnp.abs(w_cf / w_gh - 1))))
            rel_gh_nocorr.append(float(jnp.max(jnp.abs(w_nc / w_gh - 1))))
            corr.append(float(jnp.max(jnp.abs(w_cf / w_nc - 1))))
            # Monte Carlo over (z', K') with the law of motion's variance
            Km, Kv = glom.predict_next_K(lom, jnp.stack([a.z, feats[t, e, 1], jnp.float32(T - t)])[None])
            key, k1, k2 = jax.random.split(key, 3)
            zs = env.params.rho * a.z + env.params.nu * jax.random.normal(k1, (cli.mc,))
            Ks = Km[0] + jnp.sqrt(Kv[0]) * jax.random.normal(k2, (cli.mc,))
            xs = jnp.stack([zs, Ks, jnp.full((cli.mc,), tl, zs.dtype)], -1)
            w_mc = gb.log_ce_to_value(gb.gp_predict(gp, xs), tl, gamma, sigma).mean(0)
            w_cf2 = glom.continuation(env, a, Km[0], Kv[0], gp, gamma)
            rel_mc.append(float(jnp.max(jnp.abs(w_cf2 / w_mc - 1))))
    out["closed_form_vs_gauss_hermite40_K_exact_max_rel_pct"] = 100 * max(rel_gh)
    out["gauss_hermite5_vs_gauss_hermite40_K_exact_max_rel_pct"] = 100 * max(rel_gh5)
    out["closed_form_without_lognormal_correction_vs_gauss_hermite40_max_rel_pct"] = 100 * max(rel_gh_nocorr)
    out["lognormal_correction_max_rel_pct"] = 100 * max(corr)
    out["closed_form_vs_monte_carlo_K_uncertain_max_rel_pct"] = 100 * max(rel_mc)
    out["monte_carlo_draws"] = cli.mc

    # --- 4. whole backward pass at the stations: exact K' vs law of motion + closed form ---
    Z, K, L = glom.support_from_paths(env, stations)
    template = gb._period(stations, 0)
    glom.backward_pass_single(env, template, Z, K, L, lom, gamma)  # compile
    t0 = time.perf_counter()
    _, best_lom, val_lom, _ = glom.backward_pass_single(env, template, Z, K, L, lom, gamma)
    jax.block_until_ready(val_lom)
    time_lom = time.perf_counter() - t0
    gb.backward_pass_single(env, stations, gamma)
    t0 = time.perf_counter()
    gb.backward_pass_single(env, stations, gamma)
    time_exact = time.perf_counter() - t0
    w = np.asarray(stations.mu[: T - 1]) / np.asarray(stations.mu[: T - 1]).sum(-1, keepdims=True)
    d_share = np.abs(np.asarray(best_lom) - np.asarray(best_ex))
    d_val = np.abs(np.asarray(val_lom) / np.asarray(val_ex) - 1)
    ce = lambda r: (1 + r) ** (1 / (1 - sigma)) - 1  # noqa: E731  value ratio -> consumption equivalent
    out["backward_pass"] = dict(
        share_abs_diff_pop_weighted=float((w * d_share).sum(-1).mean()),
        share_abs_diff_max=float(d_share.max()),
        value_ce_diff_pop_weighted_pct=float(100 * np.abs(ce((w * d_val).sum(-1).mean()))),
        value_ce_diff_max_pct=float(100 * np.abs(ce(d_val.max()))),
        time_exact_s=time_exact,
        time_law_of_motion_s=time_lom,
    )
    print(json.dumps(out, indent=1), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / cli.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
