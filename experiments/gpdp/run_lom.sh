#!/bin/bash
# The report configuration (single value GP, alpha = 1, 20 iterations, 5 seeds) with a GP law of motion
# for K and the closed-form expectation over (z', K') (gpdp_lom.py):
#   lom_s*        Bellman equation solved at the stations' (z, K, t); K' from the law of motion
#   lom_paths8_s* solved at 8 support paths simulated with the law of motion itself (no push-forward)
# Then the held-out evaluation and relative errors, as for the report runs.
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_lom.sh
set -e
PY=${PY:-python}
LOGS=experiments/gpdp/results/logs
R=runs/endogenous
mkdir -p "$LOGS"
run() {  # run <log name> <command...>; a training run whose log.json exists is skipped (resume)
    local name=$(echo "${@:2}" | sed -n 's/.*--run-name \([^ ]*\).*/\1/p')
    if [ -n "$name" ] && [ -f "$R/gpdp/$name/log.json" ]; then echo "=== $1  skipped (done)"; return; fi
    echo "=== $1  ($(date +%H:%M:%S))"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    done in $(( $(date +%s) - t0 ))s"
}
GPDP="$PY -m mfax.algos.hsm.algos.gpdp --single-vgp --alpha 1.0 --num-iterations 20 --law-of-motion"
for s in 0 1 2 3 4; do run gpdp_lom_s$s $GPDP --seed $s --run-name lom_s$s; done
for s in 0 1 2 3 4; do run gpdp_lom_paths8_s$s $GPDP --lom-support-paths 8 --seed $s --run-name lom_paths8_s$s; done

A=""; B=""; for s in 0 1 2 3 4; do A="$A $R/gpdp/lom_s$s"; B="$B $R/gpdp/lom_paths8_s$s"; done
run lom_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $A $B --out lom_evaluate.json
run lom_relative_errors $PY experiments/gpdp/relative_errors.py --runs $A $B --out relative_errors_lom.json
echo "all done"
