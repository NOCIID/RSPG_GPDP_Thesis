#!/bin/bash
# Check 1 (FINDINGS.md): GPDP trained on MFAX's 20 grid actions, then a held-out evaluation of all
# methods (grid GPDP, continuous GPDP, original RSPG, RSPG + GP critic) on the same metrics.
# Needs run_step4.sh's GPDP runs and run_all.sh's base_s*, gpfix_s*.
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_check1_grid.sh
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
GPDP="$PY -m mfax.algos.hsm.algos.gpdp --grid-actions"

run gpdp_grid_const15_a05_s0 $GPDP --run-name grid_const15_a05_s0
for s in 1 2 3 4; do run gpdp_grid_const15_a05_s$s $GPDP --seed $s --run-name grid_const15_a05_s$s --eval-frequency 5; done
run gpdp_grid_const15_a1_s0 $GPDP --alpha 1.0 --run-name grid_const15_a1_s0

R=runs/endogenous
RUNS=""
for s in 0 1 2 3 4; do RUNS="$RUNS $R/gpdp/grid_const15_a05_s$s"; done
RUNS="$RUNS $R/gpdp/grid_const15_a1_s0"
for s in 0 1 2 3 4; do RUNS="$RUNS $R/gpdp/const15_a05_s$s"; done
RUNS="$RUNS $R/gpdp/const15_a1_s0"
for m in base gpfix; do for s in 0 1 2 3 4; do RUNS="$RUNS $R/hsm_rspg_gp/${m}_s$s"; done; done
run check1_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $RUNS --out check1_evaluate.json
echo "all done"
