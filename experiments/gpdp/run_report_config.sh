#!/bin/bash
# The GPDP configuration described in reports/gpdp_report.tex: one value GP over (z, K, T - t) for all
# periods, no blending (alpha = 1), policy GP over (z, K, T - t), continuous shares. 5 seeds,
# evaluated every iteration (for convergence times), then the held-out evaluation and relative errors.
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_report_config.sh
set -e
PY=${PY:-python}
LOGS=experiments/gpdp/results/logs
R=runs/endogenous
mkdir -p "$LOGS"
run() {  # run <log name> <command...>
    echo "=== $1  ($(date +%H:%M:%S))"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    done in $(( $(date +%s) - t0 ))s"
}
GPDP="$PY -m mfax.algos.hsm.algos.gpdp --single-vgp --alpha 1.0 --num-iterations 20"
for s in 0 1 2 3 4; do run gpdp_report_s$s $GPDP --seed $s --run-name report_singlevgp_a1_s$s; done

NEW=""; for s in 0 1 2 3 4; do NEW="$NEW $R/gpdp/report_singlevgp_a1_s$s"; done
run report_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $NEW --out report_evaluate.json
run report_relative_errors $PY experiments/gpdp/relative_errors.py --runs $NEW --out relative_errors_report.json
echo "all done"
