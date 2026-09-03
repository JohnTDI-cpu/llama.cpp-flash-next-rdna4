#!/usr/bin/env bash
# Runs the three E0 any-order microbenchmarks in order on ROCR_VISIBLE_DEVICES (default: 1).
# Binaries must be compiled first, e.g.:
#   /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -Wall -o e0b_eventsync  e0b_eventsync.hip
#   /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -Wall -o e0c_copies     e0c_copies.hip
#   /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -Wall -o e0f_launchcost e0f_launchcost.hip
cd "$(dirname "$0")" || exit 1
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-1}"
unset HIP_VISIBLE_DEVICES
LOG="$PWD/run_e0.log"
: > "$LOG"
echo "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES" | tee -a "$LOG"
for b in e0b_eventsync e0c_copies e0f_launchcost; do
    echo "=== $b ===" | tee -a "$LOG"
    if [ ! -x "./$b" ]; then
        echo "missing binary ./$b (compile it first)" | tee -a "$LOG"
        echo "rc=127" | tee -a "$LOG"
        continue
    fi
    timeout -s KILL 300 "./$b" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "rc=$rc" | tee -a "$LOG"
done
echo
echo "=== SUMMARY (PASS/FAIL, OVERTAKE, HOST ENQUEUE, rc) ==="
grep -E '(PASS|FAIL)[[:space:]]*$|OVERTAKE:|HOST ENQUEUE|^rc=|^=== ' "$LOG"
