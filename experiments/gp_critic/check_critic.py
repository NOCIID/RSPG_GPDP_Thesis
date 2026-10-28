"""
Check 2: is the GP critic accurate, and do lagged prices help?
Check 3 (RSPG only): how much does the trained policy use its memory?

2a. Critic vs brute-force Monte Carlo. Test states are drawn where the critic is used in training:
    at the end of the windows (periods H, 2H, 3H) of fresh episodes under the trained policy.
    For each test state, the "true" value of every individual state is the average of M independent
    continuations (new aggregate shocks), evaluated by exact backward recursion:
    - reference "infinite": L periods with no terminal period (what an infinite-horizon critic
      estimates);
    - reference "finite": until the end of the 128-period episode, with its terminal reward (the
      quantity the training objective actually needs, and what a time-aware critic estimates).
    The default is the critic's own definition.
    Errors are reported in consumption-equivalent (CE) units: with CRRA utility, the constant
    consumption c_bar giving the same value satisfies V = K u(c_bar), so
        c_bar_GP / c_bar_MC - 1 = (V_GP / V_MC)^(1 / (1 - sigma)) - 1     (K cancels).
    Population-weighted errors use the distribution mu at the test state (the weights the critic
    receives in the objective); "visited" maxima are over states with mu > 1e-4.

2b. Lags. GPs with 0, 1 and 2 lagged (z, K) are fitted on *independent* targets (one target per
    simulated economy, single continuation), then scored on the Monte Carlo test set of 2a.
    Each GP is fitted from two starting length scales (1 and 3) and the fit with the better
    marginal likelihood is kept (model selection on training data only).

3.  Memory. At the test states, the expected consumption share E[a | x] is computed with the true
    GRU state, with a zero GRU state (no memory), and with the GRU state of another economy.

    python experiments/gp_critic/check_critic.py --run-dir runs/endogenous/hsm_rspg_gp/gp_s0
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

from mfax.algos.hsm.utils.gp_critic import (  # noqa: E402
    GPHyper,
    fit_gp,
    gp_inputs,
    gp_predict,
    init_hyper,
    n_base_inputs,
)
from mfax.algos.hsm.utils.gp_rollouts import (  # noqa: E402
    MFPolicyFn,
    forward_rollout,
    reset_batch,
    value_rollout,
)
from mfax.envs import make_env  # noqa: E402

MAX_LAGS = 2


def tile(tree, m):
    """Repeat each economy m times along the leading axis (economy-major)."""
    return jax.tree.map(lambda x: jnp.repeat(x, m, axis=0), tree)


def test_states(env, policy_fn, params, key, times, per_time):
    """Fresh episodes from the reset, simulated to each time in `times` (per_time economies each)."""
    out = []
    for t in times:
        key, k_reset, k_roll = jax.random.split(key, 3)
        s, obs, h, fh = reset_batch(env, k_reset, per_time, MAX_LAGS, policy_fn)
        (s, obs, h, fh), _ = forward_rollout(env, policy_fn, params, s, obs, h, fh, k_roll, length=t)
        out.append((s, obs, h, fh))
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *out)


def inputs(state, time_input, T):
    s, _, _, fh = state
    return np.asarray(gp_inputs(fh, T - s.time if time_input else None))


def mc_values(env, policy_fn, params, gamma, state, key, m, length, chunk, finite):
    """Mean and standard error of V_0 over m continuations for each test state."""
    s, obs, h, fh = state
    n = obs.shape[0]
    tiled = tile((s, obs, h, fh), m)
    vals = []
    for i in range(0, n * m, chunk):
        key, k = jax.random.split(key)
        part = jax.tree.map(lambda x: x[i : i + chunk], tiled)
        v0, _, _ = value_rollout(
            env, policy_fn, params, gamma, *part, k, length=length, return_all=False, finite=finite
        )
        vals.append(np.asarray(v0))
    vals = np.concatenate(vals).reshape(n, m, -1)
    return vals.mean(1), vals.std(1, ddof=1) / np.sqrt(m)


def ce_ratio(v_hat, v_ref, sigma):
    return (v_hat / v_ref) ** (1.0 / (1.0 - sigma)) - 1.0


def score(v_hat, v_mc, se_mc, mu, sigma):
    """Accuracy of predicted values (n_test, n_states) against the Monte Carlo reference."""
    err = ce_ratio(v_hat, v_mc, sigma)
    noise = np.abs(ce_ratio(v_mc + se_mc, v_mc, sigma))  # CE size of one MC standard error
    w = mu / mu.sum(1, keepdims=True)
    visited = mu > 1e-4
    # --- R^2 over the aggregate variation: centre each individual state across test states ---
    resid = ((v_hat - v_mc) ** 2).sum()
    total = ((v_mc - v_mc.mean(0, keepdims=True)) ** 2).sum()
    noise_var = (se_mc**2).sum()
    return dict(
        ce_err_pop_weighted_mean=float(np.mean((w * np.abs(err)).sum(1))),
        ce_err_signed_pop_weighted_mean=float(np.mean((w * err).sum(1))),
        ce_err_unweighted_mean=float(np.abs(err).mean()),
        ce_err_visited_max=float(np.abs(err)[visited].max()),
        ce_err_all_states_max=float(np.abs(err).max()),
        rel_value_err_pop_weighted_mean=float(
            np.mean((w * np.abs((v_hat - v_mc) / v_mc)).sum(1))
        ),
        mc_noise_ce_pop_weighted_mean=float(np.mean((w * noise).sum(1))),
        r2=float(1 - resid / total),
        r2_ceiling_given_mc_noise=float(1 - noise_var / total),
    )


def independent_targets(env, policy_fn, params, gamma, key, times, per_time, length, finite, time_input):
    """One target per economy: simulate to t*, then a single continuation."""
    T = env.params.max_steps_in_episode
    Xs, Vs = [], []
    for t in times:
        key, k_reset, k_roll, k_val = jax.random.split(key, 4)
        s, obs, h, fh = reset_batch(env, k_reset, per_time, MAX_LAGS, policy_fn)
        state, _ = forward_rollout(env, policy_fn, params, s, obs, h, fh, k_roll, length=t)
        v0, x0, _ = value_rollout(
            env, policy_fn, params, gamma, *state, k_val,
            length=(T - t) if finite else length, return_all=False,
            finite=finite, time_input=time_input,
        )
        Xs.append(np.asarray(x0))
        Vs.append(np.asarray(v0))
    return np.concatenate(Xs), np.concatenate(Vs)


def fit_best(X, V, n_base, starts=(1.0, 3.0)):
    """Fit from several starting length scales for the base inputs; keep the best NLML."""
    best = None
    for ell0 in starts:
        hyp = init_hyper(X.shape[1], n_base)
        hyp = GPHyper(
            log_ell=hyp.log_ell.at[:n_base].set(jnp.log(ell0)),
            log_sf=hyp.log_sf,
            log_sn=hyp.log_sn,
        )
        g, nll = fit_gp(jnp.asarray(X), jnp.asarray(V), hyp, 400, 0.05)
        if best is None or float(nll) < best[1]:
            best = (g, float(nll), ell0)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--reference", choices=["auto", "finite", "infinite"], default="auto")
    p.add_argument("--per-time", type=int, default=8, help="test economies per test period")
    p.add_argument("--mc", type=int, default=32, help="continuations per test state")
    p.add_argument("--mc-length", type=int, default=170, help="periods (infinite reference)")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--n-independent", type=int, default=200, help="independent targets per period")
    p.add_argument("--skip-lags", action="store_true")
    p.add_argument("--seed", type=int, default=2024)
    cli = p.parse_args()

    run_dir = Path(cli.run_dir)
    run_args = json.loads((run_dir / "args.json").read_text())
    with open(run_dir / "mf_agent_wrapper.pkl", "rb") as f:
        wrapper = pickle.load(f)
    with open(run_dir / "gp_state.pkl", "rb") as f:
        gp = pickle.load(f)
    env = make_env("pushforward/endogenous", common_noise=True)
    T = env.params.max_steps_in_episode
    policy_fn = MFPolicyFn.from_wrapper(env, wrapper, run_args["state_type"])
    params = wrapper.mf_params
    gamma = run_args["discount_factor"]
    sigma = env.params.sigma
    time_input = run_args.get("time_input", False)
    n_base = n_base_inputs(time_input)
    reference = cli.reference if cli.reference != "auto" else ("finite" if time_input else "infinite")
    finite = reference == "finite"
    H = run_args["horizon"]
    times = (H, 2 * H, 3 * H)
    out = dict(
        run_dir=str(run_dir), times=times, mc=cli.mc, reference=reference, time_input=time_input,
        mc_length=(f"until period {T}" if finite else cli.mc_length),
    )

    # --- 2a: training critic vs Monte Carlo ---
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(cli.seed)
    key, k_test, k_mc = jax.random.split(key, 3)
    state = test_states(env, policy_fn, params, k_test, times, cli.per_time)
    s, obs, h, fh = state
    mu = np.asarray(s.mu)
    length = T - min(times) if finite else cli.mc_length
    v_mc, se_mc = mc_values(env, policy_fn, params, gamma, state, k_mc, cli.mc, length, cli.chunk, finite)
    x_test = inputs(state, time_input, T)  # (n, n_base + 2 * MAX_LAGS)
    n_inputs = gp.X.shape[1]
    v_gp = np.asarray(gp_predict(gp, jnp.asarray(x_test[:, :n_inputs])))
    out["mc_time_s"] = time.perf_counter() - t0
    out["training_critic"] = score(v_gp, v_mc, se_mc, mu, sigma)
    out["training_critic"]["n_inputs"] = int(n_inputs)
    out["training_critic"]["by_test_period"] = {
        int(t): score(
            v_gp[i * cli.per_time : (i + 1) * cli.per_time],
            v_mc[i * cli.per_time : (i + 1) * cli.per_time],
            se_mc[i * cli.per_time : (i + 1) * cli.per_time],
            mu[i * cli.per_time : (i + 1) * cli.per_time],
            sigma,
        )["ce_err_signed_pop_weighted_mean"]
        for i, t in enumerate(times)
    }
    out["test_K_range"] = [float(x_test[:, 1].min()), float(x_test[:, 1].max())]
    out["test_z_range"] = [float(x_test[:, 0].min()), float(x_test[:, 0].max())]
    out["value_range"] = [float(v_mc.min()), float(v_mc.max())]
    print("training critic:", json.dumps(out["training_critic"], indent=2), flush=True)

    # --- 2b: lags, fitted on independent targets ---
    if not cli.skip_lags:
        t0 = time.perf_counter()
        key, k_ind = jax.random.split(key)
        X_ind, V_ind = independent_targets(
            env, policy_fn, params, gamma, k_ind, times + (min(4 * H, T - 16),),
            cli.n_independent, cli.mc_length, finite, time_input,
        )
        out["independent_targets"] = int(X_ind.shape[0])
        out["lags"] = {}
        for n_lags in range(MAX_LAGS + 1):
            d = n_base + 2 * n_lags
            g, nll, ell0 = fit_best(X_ind[:, :d], V_ind, n_base)
            pred = np.asarray(gp_predict(g, jnp.asarray(x_test[:, :d])))
            res = score(pred, v_mc, se_mc, mu, sigma)
            res["nlml"] = nll
            res["initial_length_scale"] = ell0
            res["length_scales"] = np.exp(np.asarray(g.hyp.log_ell)).round(3).tolist()
            out["lags"][n_lags] = res
            print(f"lags={n_lags}:", json.dumps(res, indent=2), flush=True)
        out["lags_time_s"] = time.perf_counter() - t0

    # --- 3: memory use (RSPG) ---
    if policy_fn.recurrent:
        done = jnp.zeros((obs.shape[0],), dtype=jnp.bool_)

        def mean_a(hidden):
            prob_a, _ = policy_fn(params, hidden, obs, done)
            return np.asarray(policy_fn.mean_action(prob_a))

        a_true = mean_a(h)
        a_zero = mean_a(jnp.zeros_like(h))
        a_other = mean_a(jnp.roll(h, 1, axis=0))
        w = mu / mu.sum(1, keepdims=True)

        def gap(a):
            rel = np.abs(a / a_true - 1)
            return dict(
                pop_weighted_mean_rel=float(np.mean((w * rel).sum(1))),
                visited_max_rel=float(rel[mu > 1e-4].max()),
            )

        across = np.abs(a_true / a_true.mean(0, keepdims=True) - 1)
        out["memory"] = dict(
            zero_memory=gap(a_zero),
            other_economy_memory=gap(a_other),
            variation_across_test_states=dict(
                pop_weighted_mean_rel=float(np.mean((w * across).sum(1)))
            ),
            mean_consumption_share=float(np.mean((w * a_true).sum(1))),
        )
        print("memory:", json.dumps(out["memory"], indent=2), flush=True)

    default_ref = "finite" if time_input else "infinite"
    suffix = "" if reference == default_ref else f"_{reference}"
    name = f"critic_{run_dir.parent.name}_{run_dir.name}{suffix}.json"
    if run_dir.name.startswith("ckpt_"):
        name = f"critic_{run_dir.parent.parent.name}_{run_dir.parent.name}_{run_dir.name}{suffix}.json"
    path = Path(__file__).parent / "results" / name
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
