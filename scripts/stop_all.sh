#!/usr/bin/env bash
# Gracefully stop sensor collectors on all machines.
#
# Sends SIGTERM to the collector process, which triggers graceful shutdown
# (final CSV flush, metadata update).

set -euo pipefail

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

stop_direct() {
    local name="$1"
    local ssh_cmd="$2"

    log "Stopping collector on $name..."
    $ssh_cmd "
        if [ -f ~/drift_data/collector.pid ]; then
            pid=\$(cat ~/drift_data/collector.pid)
            if kill -0 \$pid 2>/dev/null; then
                kill -TERM \$pid
                # Wait up to 10s for graceful shutdown
                for i in \$(seq 1 10); do
                    kill -0 \$pid 2>/dev/null || break
                    sleep 1
                done
                if kill -0 \$pid 2>/dev/null; then
                    echo '  Warning: process still running, sending SIGKILL'
                    kill -9 \$pid 2>/dev/null || true
                fi
            else
                echo '  Process already stopped'
            fi
            rm -f ~/drift_data/collector.pid
        else
            echo '  No PID file found'
        fi
    " 2>/dev/null || log "  Warning: could not reach $name"
    log "  $name: stopped"
}

stop_jump() {
    local jump="$1"
    local target="$2"

    log "Stopping collector on $target via $jump..."
    ssh "$jump" "ssh $target '
        if [ -f ~/drift_data/collector.pid ]; then
            pid=\$(cat ~/drift_data/collector.pid)
            if kill -0 \$pid 2>/dev/null; then
                kill -TERM \$pid
                for i in \$(seq 1 10); do
                    kill -0 \$pid 2>/dev/null || break
                    sleep 1
                done
                if kill -0 \$pid 2>/dev/null; then
                    kill -9 \$pid 2>/dev/null || true
                fi
            fi
            rm -f ~/drift_data/collector.pid
        fi
    '" 2>/dev/null || log "  Warning: could not reach $target"
    log "  $target: stopped"
}

# Also stop stress scripts
log "Stopping stress scripts..."
ssh chameleon "pkill -f stress-ng 2>/dev/null; pkill -f stress_cycle 2>/dev/null" 2>/dev/null || true
ssh ares "ssh ares-comp-10 'pkill -f stress_cycle.py 2>/dev/null'" 2>/dev/null || true

stop_direct "homelab" "ssh homelab"
stop_direct "chameleon" "ssh chameleon"
stop_direct "ares" "ssh ares"
stop_jump "ares" "ares-comp-10"

log "All collectors stopped."
