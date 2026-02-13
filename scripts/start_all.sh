#!/usr/bin/env bash
# Start sensor collector on all machines with coordinated timing.
#
# Usage: ./start_all.sh [duration_seconds]
#   Default: 0 (unlimited, stop with stop_all.sh or Ctrl+C)

set -euo pipefail

DURATION="${1:-0}"
REMOTE_DIR="sensor_collector"
PYTHON="python3"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

# Machine definitions: name, ssh_command, extra_args
# Direct machines
declare -A MACHINES
MACHINES[homelab]="ssh homelab"
MACHINES[chameleon]="ssh chameleon"
MACHINES[ares]="ssh ares"

start_direct() {
    local name="$1"
    local ssh_cmd="$2"
    local extra_args="${3:-}"

    log "Starting collector on $name..."
    $ssh_cmd "cd ~/$REMOTE_DIR && nohup $PYTHON -m sensor_collector \
        -d $DURATION $extra_args \
        > ~/drift_data/collector.log 2>&1 &
        echo \$! > ~/drift_data/collector.pid"
    log "  $name: started"
}

start_jump() {
    local jump="$1"
    local target="$2"
    local extra_args="${3:-}"

    log "Starting collector on $target via $jump..."
    ssh "$jump" "ssh $target 'cd ~/$REMOTE_DIR && nohup $PYTHON -m sensor_collector \
        -d $DURATION $extra_args \
        > ~/drift_data/collector.log 2>&1 &
        echo \$! > ~/drift_data/collector.pid'"
    log "  $target: started"
}

# Ensure output dirs exist
for name in homelab chameleon ares; do
    ${MACHINES[$name]} "mkdir -p ~/drift_data" 2>/dev/null || true
done
ssh ares "ssh ares-comp-10 'mkdir -p ~/drift_data'" 2>/dev/null || true

# Start all collectors
# homelab: full collector, root sensors, no stress
start_direct "homelab" "ssh homelab" ""

# chameleon: full collector, root sensors, stressed externally
start_direct "chameleon" "ssh chameleon" ""

# ares master: lesser collector, no root
start_direct "ares" "ssh ares" "--no-root-sensors"

# ares-comp-10: lesser collector, no root, stressed externally
start_jump "ares" "ares-comp-10" "--no-root-sensors"

log ""
log "All collectors started."
log "Duration: $([ "$DURATION" = "0" ] && echo "unlimited" || echo "${DURATION}s")"
log "Use stop_all.sh to stop, or wait for duration to elapse."
