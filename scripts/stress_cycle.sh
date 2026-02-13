#!/usr/bin/env bash
# Stress cycle orchestrator for chameleon (stress-ng).
#
# Cycles through CPU, memory, CPU+mem, and I/O stress modes with idle
# cool-down periods. One full cycle = 4 hours, repeats 6 times = 24 hours.
#
# Usage: ./stress_cycle.sh [num_cycles]
#   Default: 6 cycles (24 hours)
#
# Requires: stress-ng (sudo apt install stress-ng)

set -euo pipefail

NUM_CYCLES="${1:-6}"
NCPUS=$(nproc)
STRESS_DURATION=2700    # 45 minutes in seconds
IDLE_DURATION=900       # 15 minutes in seconds
TMPDIR_STRESS="/tmp/stress_io"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

cleanup() {
    log "Cleaning up..."
    rm -rf "$TMPDIR_STRESS"
    # Kill any remaining stress-ng
    pkill -f stress-ng 2>/dev/null || true
    log "Done."
}

trap cleanup EXIT

idle_phase() {
    log "=== IDLE (cool-down) for $((IDLE_DURATION / 60)) min ==="
    sleep "$IDLE_DURATION"
}

cpu_stress() {
    log "=== CPU STRESS ($NCPUS cores, 100%) for $((STRESS_DURATION / 60)) min ==="
    stress-ng --cpu "$NCPUS" --cpu-load 100 --timeout "${STRESS_DURATION}s" \
        --metrics-brief 2>&1 | while IFS= read -r line; do log "  stress-ng: $line"; done
}

mem_stress() {
    log "=== MEMORY STRESS for $((STRESS_DURATION / 60)) min ==="
    # Use 75% of total memory across workers
    local mem_total_mb
    mem_total_mb=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
    local per_worker_mb=$(( mem_total_mb * 75 / 100 / NCPUS ))
    stress-ng --vm "$NCPUS" --vm-bytes "${per_worker_mb}M" --vm-method all \
        --timeout "${STRESS_DURATION}s" --metrics-brief 2>&1 | \
        while IFS= read -r line; do log "  stress-ng: $line"; done
}

cpu_mem_stress() {
    log "=== CPU + MEMORY STRESS for $((STRESS_DURATION / 60)) min ==="
    local mem_total_mb
    mem_total_mb=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
    local per_worker_mb=$(( mem_total_mb * 50 / 100 / NCPUS ))
    stress-ng --cpu "$((NCPUS / 2))" --cpu-load 100 \
        --vm "$((NCPUS / 2))" --vm-bytes "${per_worker_mb}M" \
        --timeout "${STRESS_DURATION}s" --metrics-brief 2>&1 | \
        while IFS= read -r line; do log "  stress-ng: $line"; done
}

io_stress() {
    log "=== I/O STRESS for $((STRESS_DURATION / 60)) min ==="
    mkdir -p "$TMPDIR_STRESS"
    stress-ng --hdd 4 --hdd-bytes 1G --temp-path "$TMPDIR_STRESS" \
        --timeout "${STRESS_DURATION}s" --metrics-brief 2>&1 | \
        while IFS= read -r line; do log "  stress-ng: $line"; done
    rm -rf "$TMPDIR_STRESS"
}

# Main loop
log "Starting stress cycle: $NUM_CYCLES cycles, $NCPUS CPUs"
log "Each cycle: 4 hours (45min stress + 15min idle × 4 modes)"

for cycle in $(seq 1 "$NUM_CYCLES"); do
    log ">>> CYCLE $cycle / $NUM_CYCLES <<<"

    cpu_stress
    idle_phase

    mem_stress
    idle_phase

    cpu_mem_stress
    idle_phase

    io_stress
    idle_phase

    log "<<< CYCLE $cycle COMPLETE >>>"
done

log "All $NUM_CYCLES cycles complete."
