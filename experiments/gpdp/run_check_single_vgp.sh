#!/bin/bash
# One value GP over (z, K, T - t) for all periods (--single-vgp), against the default one value GP per
# period (FINDINGS.md). Needs run_step4.sh's const15_a05_s* runs.
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_check_single_vgp.sh
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
GPDP="$PY -m mfax.algos.hsm.algos.gpdp --single-vgp"

run gpdp_singlevgp_const15_a05_s0 $GPDP --run-name singlevgp_const15_a05_s0
for s in 1 2 3 4; do run gpdp_singlevgp_const15_a05_s$s $GPDP --seed $s --run-name singlevgp_const15_a05_s$s --eval-frequency 5; done

R=runs/endogenous/gpdp
NEW=""; for s in 0 1 2 3 4; do NEW="$NEW $R/singlevgp_const15_a05_s$s"; done
OLD=""; for s in 0 1 2 3 4; do OLD="$OLD $R/const15_a05_s$s"; done
run check_singlevgp_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $NEW --out check_singlevgp_evaluate.json
run check_singlevgp_relative_errors $PY experiments/gpdp/relative_errors.py --runs $NEW $OLD --out relative_errors_singlevgp.json
echo "all done"
