#!/bin/bash
# Time-input ablation (FINDINGS.md): GPDP whose policy GP takes (z, K) only, without the time left,
# against the default (z, K, T - t). Needs run_step4.sh's const15_a05_s* runs.
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_check_time.sh
set -e
PY=${PY:-python}
LOGS=experiments/gpdp/results/logs
mkdir -p "$LOGS"
run() {  # run <log name> <command...>
    echo "=== $1  ($(date +%H:%M:%S))"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    done in $(( $(date +%s) - t0 ))s"
}
GPDP="$PY -m mfax.algos.hsm.algos.gpdp --no-policy-time"

run gpdp_notime_const15_a05_s0 $GPDP --run-name notime_const15_a05_s0
for s in 1 2 3 4; do run gpdp_notime_const15_a05_s$s $GPDP --seed $s --run-name notime_const15_a05_s$s --eval-frequency 5; done

R=runs/endogenous/gpdp
NEW=""; for s in 0 1 2 3 4; do NEW="$NEW $R/notime_const15_a05_s$s"; done
OLD=""; for s in 0 1 2 3 4; do OLD="$OLD $R/const15_a05_s$s"; done
run check_time_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $NEW --out check_time_evaluate.json
run check_time_relative_errors $PY experiments/gpdp/relative_errors.py --runs $NEW $OLD --out relative_errors_time.json
echo "all done"
