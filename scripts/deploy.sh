#!/usr/bin/env bash
# Deploy sensor_collector to all remote machines.
#
# Copies the src/ directory and pyproject.toml to each machine at
# ~/sensor_collector/. No install needed — runs with `python3 -m sensor_collector`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MACHINES=("homelab" "chameleon")
JUMP_MACHINES=("ares:ares-comp-10")  # jump_host:target

REMOTE_DIR="sensor_collector"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

deploy_direct() {
    local host="$1"
    log "Deploying to $host..."
    ssh "$host" "mkdir -p ~/$REMOTE_DIR/src"
    scp -r "$PROJECT_DIR/src/sensor_collector" "$host:~/$REMOTE_DIR/src/"
    scp "$PROJECT_DIR/pyproject.toml" "$host:~/$REMOTE_DIR/"
    log "  $host: done"
}

deploy_jump() {
    local jump="$1"
    local target="$2"
    log "Deploying to $target via $jump..."

    # First deploy to jump host
    ssh "$jump" "mkdir -p /tmp/sensor_deploy/src"
    scp -r "$PROJECT_DIR/src/sensor_collector" "$jump:/tmp/sensor_deploy/src/"
    scp "$PROJECT_DIR/pyproject.toml" "$jump:/tmp/sensor_deploy/"

    # Then from jump host to target
    ssh "$jump" "
        ssh $target 'mkdir -p ~/$REMOTE_DIR/src'
        scp -r /tmp/sensor_deploy/src/sensor_collector $target:~/$REMOTE_DIR/src/
        scp /tmp/sensor_deploy/pyproject.toml $target:~/$REMOTE_DIR/
        rm -rf /tmp/sensor_deploy
    "
    log "  $target (via $jump): done"
}

# Also deploy to ares master itself (direct SSH, lesser collector)
MACHINES+=("ares")

for host in "${MACHINES[@]}"; do
    deploy_direct "$host"
done

for entry in "${JUMP_MACHINES[@]}"; do
    IFS=':' read -r jump target <<< "$entry"
    deploy_jump "$jump" "$target"
done

log "Deployment complete."
