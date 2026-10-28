#!/bin/bash
# Phase layout (6, 2) against the default (4, 2, 1, 1) and the original method (FINDINGS.md,
# 2026-10-12). Needs the base_s*, gpfix_s* runs from run_all.sh. About 1h45 on an M5 Pro CPU.
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gp_critic/run_gp62.sh
set -e
PY=${PY:-python}
LOGS=experiments/gp_critic/results/logs
X=experiments/gp_critic
R=runs/endogenous
mkdir -p "$LOGS"

run() {  # run <log name> <command...>
    echo "=== $1  ($(date +%H:%M:%S))"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    done in $(( $(date +%s) - t0 ))s"
}

GPFIX="--use-gp --horizon 32 --phase-counts 4 2 1 1 --time-input"
GP62="--use-gp --horizon 32 --phase-counts 6 2 --time-input"
RSPG="$PY -m mfax.algos.hsm.algos.rspg_gp"
SPG="$PY -m mfax.algos.hsm.algos.spg_gp"

# --- training (sequential, for clean timings) ---
run rspg_gpfix_s0_repro $RSPG --seed 0 --run-name gpfix_s0_repro $GPFIX   # reproduction check
for s in 0 1 2 3 4; do run rspg_gp62_s$s $RSPG --seed $s --run-name gp62_s$s $GP62; done
for s in 0 1 2; do run spg_gp62_s$s $SPG --seed $s --run-name gp62_s$s $GP62; done

# --- evaluation ---
RS=""; for m in base gpfix gp62; do for s in 0 1 2 3 4; do RS="$RS $R/hsm_rspg_gp/${m}_s$s"; done; done
SS=""; for m in base gpfix gp62; do for s in 0 1 2; do SS="$SS $R/hsm_spg_gp/${m}_s$s"; done; done
run gp62_compare_rspg $PY $X/compare_runs.py --ref $R/hsm_rspg_gp/base_s0 \
    --runs ${RS/ $R\/hsm_rspg_gp\/base_s0/} $R/hsm_rspg_gp/gpfix_s0_repro --out compare_rspg_gp62.json
run gp62_compare_spg $PY $X/compare_runs.py --ref $R/hsm_spg_gp/base_s0 \
    --runs ${SS/ $R\/hsm_spg_gp\/base_s0/} --out compare_spg_gp62.json
run gp62_horizon $PY $X/check_horizon.py --runs $RS $SS --out horizon_gp62.json
run gp62_summarize_rspg $PY $X/summarize_seeds.py --compare compare_rspg_gp62.json \
    --horizon horizon_gp62.json --algo hsm_rspg_gp --out seeds_rspg_gp62.json
run gp62_summarize_spg $PY $X/summarize_seeds.py --compare compare_spg_gp62.json \
    --horizon horizon_gp62.json --algo hsm_spg_gp --out seeds_spg_gp62.json

RC=""; for m in gpfix gp62; do for s in 0 1 2 3 4; do RC="$RC $R/hsm_rspg_gp/${m}_s$s"; done; done
SC=""; for m in gpfix gp62; do for s in 0 1 2; do SC="$SC $R/hsm_spg_gp/${m}_s$s"; done; done
run gp62_critic_rspg $PY $X/check_critic_by_period.py --run-dirs $RC --out critic_by_period_rspg_gp62.json
run gp62_critic_spg $PY $X/check_critic_by_period.py --run-dirs $SC --out critic_by_period_spg_gp62.json

run gp62_gradient_rspg_gp62_s0 $PY $X/check_gradient_layouts.py \
    --run-dir $R/hsm_rspg_gp/gp62_s0 --out gradient_layouts_rspg_gp62_s0.json
run gp62_gradient_rspg_gpfix_s0 $PY $X/check_gradient_layouts.py \
    --run-dir $R/hsm_rspg_gp/gpfix_s0 --skip-initial --out gradient_layouts_rspg_gpfix_s0.json
echo "all done"
