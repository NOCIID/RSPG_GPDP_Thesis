"""
Summary table: the report configuration (report_singlevgp_a1_s*) against the same configuration with
the GP law of motion (lom_s*: Bellman equation at the stations; lom_paths8_s*: at support paths
simulated by the law of motion). Same metrics and convergence rule as reports/gpdp_report.tex.

    python experiments/gpdp/summarize_lom.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/gpdp/results"
GROUPS = {
    "report (K' exact, 5-point quadrature)": ("report_singlevgp_a1_s", "report_evaluate.json", "relative_errors_report.json"),
    "law of motion, stations": ("lom_s", "lom_evaluate.json", "relative_errors_lom.json"),
    "law of motion, dreamt support paths": ("lom_paths8_s", "lom_evaluate.json", "relative_errors_lom.json"),
}
BLOCKS = ["0-31", "32-63", "64-95", "96-111", "112-119", "120-126"]


def convergence(log, threshold=0.1):
    """Training time and iterations until the gain from deviating (clairvoyant) is below threshold %.
    Row i evaluates the policy produced by iteration i - 1, so the time is row i - 1's train_time."""
    rows = log["iterations"]
    for i, r in enumerate(rows):
        if r.get("gain_clairvoyant_pct", np.inf) < threshold and i > 0:
            return rows[i - 1]["train_time"], i
    return np.nan, np.nan


def fmt(x, d=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{d}f} ± {np.nanstd(x, ddof=1):.{d}f}"


def main():
    out = {}
    for name, (prefix, ev_file, re_file) in GROUPS.items():
        ev = json.loads((RES / ev_file).read_text())["runs"]
        re = json.loads((RES / re_file).read_text())["runs"]
        runs = [f"gpdp/{prefix}{s}" for s in range(5)]
        runs = [r for r in runs if r in ev and r in re]
        cr = [re[r]["by_best_response"]["clairvoyant_continuous"] for r in runs]
        logs = [json.loads((ROOT / "runs/endogenous" / r / "log.json").read_text()) for r in runs]
        conv = [convergence(lg) for lg in logs]
        step = [np.mean([row["step_time"] for row in lg["iterations"][1:]]) for lg in logs]
        lom_r2 = [
            [row["law_of_motion"]["r2_of_delta_K"] for row in lg["iterations"][2:] if "law_of_motion" in row]
            for lg in logs
        ]
        lom_err = [
            [row["law_of_motion"]["max_rel_err_pct"] for row in lg["iterations"][2:] if "law_of_motion" in row]
            for lg in logs
        ]
        g = dict(
            n_runs=len(runs),
            mfax_exploitability_nearest=fmt([ev[r]["mfax_nearest_grid"]["exploitability"] for r in runs]),
            gain_from_deviating_pct=fmt([c["gain_from_deviating_pct"] for c in cr], 2),
            ce_relerr_t0_pct=fmt([c["value_relerr_t0_mu0_weighted_pct"] for c in cr], 2),
            ce_relerr_all_periods_pct=fmt([c["value_relerr_all_periods_pop_weighted_pct"] for c in cr], 3),
            ce_relerr_by_block_pct=" · ".join(
                f"{np.mean([c['value_relerr_by_block_pct'][b] for c in cr]):.3f}" for b in BLOCKS
            ),
            policy_relerr_pct=fmt([c["policy_relerr_pop_weighted_pct"] for c in cr], 2),
            time_to_converge_s_median=float(np.nanmedian([c[0] for c in conv])),
            time_to_converge_s_each=[round(float(c[0]), 1) for c in conv],
            iterations_to_converge_each=[c[1] for c in conv],
            time_per_iteration_s=fmt(step, 2),
            capital_t100=fmt([ev[r]["K_mean_at"]["100"] for r in runs], 2),
            capital_t127=fmt([ev[r]["K_mean_at"]["127"] for r in runs], 2),
            non_finite_runs=sum(
                any(not np.isfinite(row.get("gain_clairvoyant_pct", 0.0)) for row in lg["iterations"]) for lg in logs
            ),
        )
        if any(lom_r2):
            g["law_of_motion_fresh_r2_min_from_it2"] = float(np.min([min(x) for x in lom_r2 if x]))
            g["law_of_motion_fresh_max_err_pct_median_from_it2"] = float(np.median([np.median(x) for x in lom_err if x]))
            g["law_of_motion_fresh_max_err_pct_worst_from_it2"] = float(np.max([max(x) for x in lom_err if x]))
        out[name] = g
    print(json.dumps(out, indent=1))
    (RES / "summary_lom.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
