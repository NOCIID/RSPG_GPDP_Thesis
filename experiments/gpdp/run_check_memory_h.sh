#!/bin/bash
# GPDP whose policy GP sees ONLY the 8-dim GRU hidden state (no (z, K), no date), as RSPG's network
# sees only its GRU's embedding. Otherwise as run_check_memory.sh (single value GP, frozen GRU from
# gpfix_h8_s*). Starts after run_check_memory.sh has finished, so the two do not compete for the CPU.
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_check_memory_h.sh
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

until [ -f experiments/gpdp/results/relative_errors_memory.json ]; do sleep 15; done
echo "previous batch done ($(date +%H:%M:%S))"

GPDP="$PY -m mfax.algos.hsm.algos.gpdp --single-vgp --no-policy-time --no-policy-memory-zk"
run gpdp_memory_hidden_only_s0 $GPDP --run-name memory_h8_only_s0 --policy-memory-run $R/hsm_rspg_gp/gpfix_h8_s0
for s in 1 2 3 4; do
    run gpdp_memory_hidden_only_s$s $GPDP --seed $s --run-name memory_h8_only_s$s --policy-memory-run $R/hsm_rspg_gp/gpfix_h8_s$s --eval-frequency 5
done

NEW=""; for s in 0 1 2 3 4; do NEW="$NEW $R/gpdp/memory_h8_only_s$s"; done
run check_memory_h_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $NEW --out check_memory_h_evaluate.json
run check_memory_h_relative_errors $PY experiments/gpdp/relative_errors.py --runs $NEW --out relative_errors_memory_h.json
echo "all done"
