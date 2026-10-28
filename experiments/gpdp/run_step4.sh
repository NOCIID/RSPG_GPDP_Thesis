#!/bin/bash
# GPDP build step 4 (FINDINGS.md, 2026-10-12): GPDP runs, then held-out evaluation of GPDP and the
# RSPG baselines (needs base_s* and gpfix_s* from experiments/gp_critic/run_all.sh).
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_step4.sh
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
GPDP="$PY -m mfax.algos.hsm.algos.gpdp"

# --- GPDP: seed 0 (evaluated every iteration), seeds 1-4, and no blending (alpha = 1) ---
[ -f runs/endogenous/gpdp/const15_a05_s0/log.json ] || run gpdp_const15_a05_s0 $GPDP --run-name const15_a05_s0
for s in 1 2 3 4; do run gpdp_const15_a05_s$s $GPDP --seed $s --run-name const15_a05_s$s --eval-frequency 5; done
run gpdp_const15_a1_s0 $GPDP --alpha 1.0 --run-name const15_a1_s0

# --- held-out evaluation: GPDP and the RSPG baselines on the same metrics ---
R=runs/endogenous
RUNS=""
for s in 0 1 2 3 4; do RUNS="$RUNS $R/gpdp/const15_a05_s$s"; done
RUNS="$RUNS $R/gpdp/const15_a1_s0"
for m in base gpfix; do for s in 0 1 2 3 4; do RUNS="$RUNS $R/hsm_rspg_gp/${m}_s$s"; done; done
run step4_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $RUNS --out step4_evaluate.json
echo "all done"
