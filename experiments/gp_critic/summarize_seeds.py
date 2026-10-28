"""
Aggregate the multi-seed runs by method: mean, standard deviation and range of each metric,
and an exact permutation test of the difference in held-out exploitability against the original.

Inputs: a compare_runs.py output (held-out evaluation, gaps to the reference) and, optionally, a
check_horizon.py output. Runs are grouped by name prefix: base_s*, gp_s*, gpfix_s*, gpfast_s*, gp62_s*.

    python experiments/gp_critic/summarize_seeds.py --compare compare_rspg_all.json \
        --horizon horizon_all.json --algo hsm_rspg_gp --out seeds_rspg.json
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
THRESHOLDS = (0.1, 0.07, 0.05, 0.042)


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs, dtype=float)
    return dict(
        mean=float(a.mean()),
        std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        min=float(a.min()),
        max=float(a.max()),
        n=len(a),
    )


def permutation_p(a, b):
    """Exact two-sided permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        count += abs(pooled[mask].mean() - pooled[~mask].mean()) >= observed - 1e-12
        total += 1
    return count / total


def time_to(curve, threshold):
    for point in curve:
        if point["exploitability"] <= threshold:
            return point["train_time"]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", required=True)
    p.add_argument("--horizon", default=None)
    p.add_argument("--algo", required=True, help="e.g. hsm_rspg_gp")
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    comp = json.loads((RESULTS / cli.compare).read_text())["runs"]
    hor = json.loads((RESULTS / cli.horizon).read_text())["runs"] if cli.horizon else {}

    groups = {}
    for name, r in comp.items():
        algo, run = name.split("/")
        m = re.fullmatch(r"(base|gp|gpfix|gpfast|gp62)_s(\d+)", run)
        if algo != cli.algo or not m:
            continue
        groups.setdefault(m.group(1), []).append((int(m.group(2)), name, r))

    out = {}
    for method, runs in sorted(groups.items()):
        runs.sort()
        rows = dict(
            seeds=[s for s, _, _ in runs],
            exploitability=stats([r["exploitability"] for _, _, r in runs]),
            consumption_gain_from_deviating=stats(
                [r["policy_return"] / r["br_return"] - 1 for _, _, r in runs]
            ),
            total_train_time_s=stats([r["total_train_time_s"] for _, _, r in runs]),
            step_time_median_s=stats([r["step_time_median_s"] for _, _, r in runs]),
            policy_gap_vs_base_s0=stats(
                [r["policy_gap_vs_ref"]["pop_weighted_mean_rel"] for _, n, r in runs if not n.endswith("base_s0")]
            ),
            K_mean_gap_vs_base_s0=stats(
                [r["aggregate_gap_vs_ref"]["K_mean_rel"] for _, n, r in runs if not n.endswith("base_s0")]
            ),
            K_at_period=dict(
                periods=runs[0][2]["aggregate_path"]["periods"],
                mean=np.mean([r["aggregate_path"]["K_mean"] for _, _, r in runs], 0).round(2).tolist(),
                std=np.std([r["aggregate_path"]["K_mean"] for _, _, r in runs], 0, ddof=1).round(2).tolist()
                if len(runs) > 1 else None,
            ),
            time_to_threshold_s={
                str(t): dict(
                    reached=sum(time_to(r["curve"], t) is not None for _, _, r in runs),
                    median=(
                        float(np.median([time_to(r["curve"], t) for _, _, r in runs if time_to(r["curve"], t) is not None]))
                        if any(time_to(r["curve"], t) is not None for _, _, r in runs) else None
                    ),
                )
                for t in THRESHOLDS
            },
        )
        if hor:
            shares = [hor[n]["consumption_share"] for _, n, _ in runs if n in hor]
            if shares:
                rows["consumption_share_by_block"] = dict(
                    mean=np.mean(shares, 0).round(4).tolist(),
                    std=np.std(shares, 0, ddof=1).round(4).tolist() if len(shares) > 1 else None,
                )
                rows["dissaving_runs"] = int(
                    sum(s[-1] > s[3] + 0.005 for s in shares)
                )  # share at 123-127 above share at 75-99
        out[method] = rows

    if "base" in groups:
        base_e = [r["exploitability"] for _, _, r in groups["base"]]
        for method in groups:
            if method != "base":
                e = [r["exploitability"] for _, _, r in groups[method]]
                out[method]["permutation_p_exploitability_vs_base"] = permutation_p(e, base_e)

    (RESULTS / cli.out).write_text(json.dumps(out, indent=2))
    for method, rows in out.items():
        print(f"== {method} (seeds {rows['seeds']})")
        for k, v in rows.items():
            if k != "seeds":
                print(f"   {k}: {v}")
    print(f"Saved {RESULTS / cli.out}")


if __name__ == "__main__":
    main()
