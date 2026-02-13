#!/usr/bin/env bash
# Transfer CSV and metadata files from all machines back to localhost.
#
# Collects into ./collected_data/<hostname>/ directory structure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/collected_data"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

collect_direct() {
    local name="$1"
    local ssh_cmd="$2"
    local dest="$OUTPUT_DIR/$name"

    mkdir -p "$dest"
    log "Collecting from $name..."

    # Get file list
    local files
    files=$($ssh_cmd "ls ~/drift_data/*.csv ~/drift_data/*.json ~/drift_data/*.log 2>/dev/null" || true)
    if [ -z "$files" ]; then
        log "  $name: no data files found"
        return
    fi

    scp "$name:~/drift_data/*.csv" "$dest/" 2>/dev/null || true
    scp "$name:~/drift_data/*.json" "$dest/" 2>/dev/null || true
    scp "$name:~/drift_data/*.log" "$dest/" 2>/dev/null || true

    local count
    count=$(ls -1 "$dest"/*.csv 2>/dev/null | wc -l)
    log "  $name: collected $count CSV file(s)"
}

collect_jump() {
    local jump="$1"
    local target="$2"
    local dest="$OUTPUT_DIR/$target"

    mkdir -p "$dest"
    log "Collecting from $target via $jump..."

    # Stage files on jump host
    ssh "$jump" "
        mkdir -p /tmp/collect_$target
        scp $target:~/drift_data/*.csv /tmp/collect_$target/ 2>/dev/null || true
        scp $target:~/drift_data/*.json /tmp/collect_$target/ 2>/dev/null || true
        scp $target:~/drift_data/*.log /tmp/collect_$target/ 2>/dev/null || true
    "

    # Pull from jump host
    scp "$jump:/tmp/collect_$target/*" "$dest/" 2>/dev/null || true
    ssh "$jump" "rm -rf /tmp/collect_$target"

    local count
    count=$(ls -1 "$dest"/*.csv 2>/dev/null | wc -l)
    log "  $target: collected $count CSV file(s)"
}

log "Collecting results to $OUTPUT_DIR"

collect_direct "homelab" "ssh homelab"
collect_direct "chameleon" "ssh chameleon"
collect_direct "ares" "ssh ares"
collect_jump "ares" "ares-comp-10"

log ""
log "Results collected to $OUTPUT_DIR"
log "Directory structure:"
find "$OUTPUT_DIR" -type f | sort | while read -r f; do
    size=$(du -h "$f" | cut -f1)
    echo "  $size  $f"
done
