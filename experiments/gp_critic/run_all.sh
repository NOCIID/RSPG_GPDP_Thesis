#!/bin/bash
# Reproduces every number in SUMMARY.md. Runs sequentially so that timings are not disturbed.
# Usage (from the repository root):  PY=python bash experiments/gp_critic/run_all.sh
# To run only some stages:           ONLY="rspg_base_s1|rspg_critic" PY=python bash ...
# Keep the machine plugged in and awake (e.g. prefix with `caffeinate -i`); about 9 hours in total.
set -e
PY=${PY:-python}
ONLY=${ONLY:-.}
R=runs/endogenous
LOGS=experiments/gp_critic/results/logs
mkdir -p "$LOGS"

run() {  # run <log name> <command...>
    if ! [[ "$1" =~ ^($ONLY)$ ]] && [ "$ONLY" != "." ]; then return 0; fi
    echo "=== $1: ${*:2}"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    done in $(( $(date +%s) - t0 ))s"
}

# --- configurations (explicit, so results do not depend on script defaults) ---
BASE="--no-use-gp --horizon 128 --phase-counts 8"                       # the original method
GP25="--use-gp --horizon 25 --phase-counts 4 2 1 1 --no-time-input"     # GP critic, first version
GPFIX="--use-gp --horizon 32 --phase-counts 4 2 1 1 --time-input"       # time-aware, whole episode
RSPG="$PY -m mfax.algos.hsm.algos.rspg_gp"
SPG="$PY -m mfax.algos.hsm.algos.spg_gp"

# --- check 0: the new train step reproduces the original one ---
run equivalence $PY experiments/gp_critic/check_equivalence.py

# ===================== RSPG =====================
# --- timing reference: the original script, unchanged ---
run orig_rspg_timing $PY -m mfax.algos.hsm.algos.rspg --num-iterations 60 --eval-frequency 20 --no-save

# --- first comparison: original vs GP critic (25-period windows), two seeds, and ablations ---
run rspg_gp_s0   $RSPG --seed 0 --run-name gp_s0 $GP25
run rspg_base_s0 $RSPG --seed 0 --run-name base_s0 $BASE
run rspg_gp_s1   $RSPG --seed 1 --run-name gp_s1 $GP25
run rspg_base_s1 $RSPG --seed 1 --run-name base_s1 $BASE
run rspg_zero_tail_s0 $RSPG --seed 0 --run-name zero_tail_s0 --no-use-gp --horizon 25 --phase-counts 4 2 1 1
run rspg_gp_reset_s0  $RSPG --seed 0 --run-name gp_reset_s0 --use-gp --horizon 25 --phase-counts 8

# --- checks on the first version ---
run rspg_critic   $PY experiments/gp_critic/check_critic.py --run-dir $R/hsm_rspg_gp/gp_s0
run rspg_gradient $PY experiments/gp_critic/check_gradient.py --run-dir $R/hsm_rspg_gp/gp_s0 \
    --variants full gp zero_tail gp_reset
run rspg_compare  $PY experiments/gp_critic/compare_runs.py --ref $R/hsm_rspg_gp/base_s0 \
    --runs $R/hsm_rspg_gp/base_s1 $R/hsm_rspg_gp/gp_s0 $R/hsm_rspg_gp/gp_s1 \
           $R/hsm_rspg_gp/zero_tail_s0 $R/hsm_rspg_gp/gp_reset_s0 --out compare_rspg.json

# ===================== SPG (first comparison) =====================
SPG_ORIG="--task endogenous --state-type states --discount-factor 0.95 --num-envs 8 --lr 0.01"
run orig_spg_timing $PY -m mfax.algos.hsm.algos.spg $SPG_ORIG --num-iterations 60 --eval-frequency 20 --no-save
run spg_gp_s0   $SPG --seed 0 --run-name gp_s0 $GP25
run spg_base_s0 $SPG --seed 0 --run-name base_s0 $BASE
run spg_critic  $PY experiments/gp_critic/check_critic.py --run-dir $R/hsm_spg_gp/gp_s0
run spg_compare $PY experiments/gp_critic/compare_runs.py --ref $R/hsm_spg_gp/base_s0 \
    --runs $R/hsm_spg_gp/gp_s0 --out compare_spg.json

# --- check 5: end-of-episode behaviour (finite vs infinite horizon) ---
run horizon $PY experiments/gp_critic/check_horizon.py --runs \
    $R/hsm_rspg_gp/base_s0 $R/hsm_rspg_gp/base_s1 $R/hsm_rspg_gp/gp_s0 $R/hsm_rspg_gp/gp_s1 \
    $R/hsm_rspg_gp/gp_reset_s0 $R/hsm_spg_gp/base_s0 $R/hsm_spg_gp/gp_s0

# --- determinism and clean timing: GP seed 0 again (in the original session, the laptop slept
#     during gp_s0, which slowed a few chunks) ---
run rspg_gp_s0_timing $RSPG --seed 0 --run-name gp_s0_timing $GP25

# ===================== Fixes (second round) =====================
# Block 1: RSPG with the time-aware critic over the whole episode (5 seeds), more seeds of the
# first GP version, and the critic checks (including checkpoints during training).
run rspg_gpfix_s0 $RSPG --seed 0 --run-name gpfix_s0 $GPFIX --checkpoint-every 50
for s in 1 2 3 4; do run rspg_gpfix_s$s $RSPG --seed $s --run-name gpfix_s$s $GPFIX; done
run rspg_gp_s2 $RSPG --seed 2 --run-name gp_s2 $GP25 --checkpoint-every 50
for s in 3 4; do run rspg_gp_s$s $RSPG --seed $s --run-name gp_s$s $GP25; done
run rspg_critic_finite_ref $PY experiments/gp_critic/check_critic.py --run-dir $R/hsm_rspg_gp/gp_s0 \
    --reference finite --skip-lags
run rspg_critic_gpfix $PY experiments/gp_critic/check_critic.py --run-dir $R/hsm_rspg_gp/gpfix_s0
for c in 50 100 250; do
    run rspg_critic_gpfix_ckpt$c $PY experiments/gp_critic/check_critic.py \
        --run-dir $R/hsm_rspg_gp/gpfix_s0/ckpt_$c --skip-lags
    run rspg_critic_gp_ckpt$c $PY experiments/gp_critic/check_critic.py \
        --run-dir $R/hsm_rspg_gp/gp_s2/ckpt_$c --skip-lags
done
run rspg_critic_gp_s2 $PY experiments/gp_critic/check_critic.py --run-dir $R/hsm_rspg_gp/gp_s2 --skip-lags
run rspg_gradient_gpfix $PY experiments/gp_critic/check_gradient.py --run-dir $R/hsm_rspg_gp/gpfix_s0 \
    --variants full gp gp_time --out gradient_rspg_gpfix.json

# Block 2: more seeds of the original RSPG.
for s in 2 3 4; do run rspg_base_s$s $RSPG --seed $s --run-name base_s$s $BASE; done

# Block 3: SPG seeds and fixes, then the multi-seed comparisons.
for s in 1 2; do run spg_base_s$s $SPG --seed $s --run-name base_s$s $BASE; done
for s in 1 2; do run spg_gp_s$s $SPG --seed $s --run-name gp_s$s $GP25; done
for s in 0 1 2; do run spg_gpfix_s$s $SPG --seed $s --run-name gpfix_s$s $GPFIX; done
run spg_critic_gpfix $PY experiments/gp_critic/check_critic.py --run-dir $R/hsm_spg_gp/gpfix_s0

# Block 4: fast early refits (the critic lags the policy early in training; one run of the first
# version collapsed). Refit after every step for the first 150 steps, on the last 5 steps' targets.
FAST="--fast-refit-steps 150 --fast-refit-window 5"
run rspg_gp_s4_fast $RSPG --seed 4 --run-name gp_s4_fast $GP25 $FAST
run rspg_gpfast_s0 $RSPG --seed 0 --run-name gpfast_s0 $GPFIX $FAST --checkpoint-every 50
for s in 1 2 3 4; do run rspg_gpfast_s$s $RSPG --seed $s --run-name gpfast_s$s $GPFIX $FAST; done
for c in 50 100; do
    run rspg_critic_gpfast_ckpt$c $PY experiments/gp_critic/check_critic.py \
        --run-dir $R/hsm_rspg_gp/gpfast_s0/ckpt_$c --skip-lags
done

# --- gradient check with each run's own trained critic (instead of a fresh warm-start fit) ---
run rspg_gradient_gpfix_runcritic $PY experiments/gp_critic/check_gradient.py \
    --run-dir $R/hsm_rspg_gp/gpfix_s0 --variants full gp_time --use-run-critic \
    --out gradient_rspg_gpfix_runcritic.json
run rspg_gradient_gp_runcritic $PY experiments/gp_critic/check_gradient.py \
    --run-dir $R/hsm_rspg_gp/gp_s0 --variants full gp --use-run-critic \
    --out gradient_rspg_gp_runcritic.json

# --- clean timing of the time-aware version on mains power (block 1 ran on battery and was
#     throttled; training is deterministic, so only the timing differs) ---
run rspg_gpfix_s0_timing $RSPG --seed 0 --run-name gpfix_s0_timing $GPFIX

RSPG_ALL=""; for m in base gp gpfix gpfast; do for s in 0 1 2 3 4; do RSPG_ALL="$RSPG_ALL $R/hsm_rspg_gp/${m}_s$s"; done; done
RSPG_ALL="$RSPG_ALL $R/hsm_rspg_gp/gp_s4_fast"
SPG_ALL="";  for m in base gp gpfix; do for s in 0 1 2; do SPG_ALL="$SPG_ALL $R/hsm_spg_gp/${m}_s$s"; done; done
run rspg_compare_all $PY experiments/gp_critic/compare_runs.py --ref $R/hsm_rspg_gp/base_s0 \
    --runs ${RSPG_ALL/ $R\/hsm_rspg_gp\/base_s0/} --out compare_rspg_all.json
run spg_compare_all $PY experiments/gp_critic/compare_runs.py --ref $R/hsm_spg_gp/base_s0 \
    --runs ${SPG_ALL/ $R\/hsm_spg_gp\/base_s0/} --out compare_spg_all.json
run horizon_all $PY experiments/gp_critic/check_horizon.py --runs $RSPG_ALL $SPG_ALL --out horizon_all.json
run summarize_rspg $PY experiments/gp_critic/summarize_seeds.py --compare compare_rspg_all.json \
    --horizon horizon_all.json --algo hsm_rspg_gp --out seeds_rspg.json
run summarize_spg $PY experiments/gp_critic/summarize_seeds.py --compare compare_spg_all.json \
    --horizon horizon_all.json --algo hsm_spg_gp --out seeds_spg.json

echo "all done"
