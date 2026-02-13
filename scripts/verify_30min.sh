#!/usr/bin/env bash
# 30-minute verification test for sensor collector with peer clock probing.
#
# Deploys latest code via git pull, starts collectors in tmux sessions on all
# 4 Linux nodes with --peers enabled, monitors progress every 60s, and prints
# a summary at the end.
#
# Usage: ./verify_30min.sh

set -euo pipefail

DURATION=1800  # 30 minutes
CHECK_INTERVAL=60
REMOTE_DIR="sensor_collector"
PYTHON="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

log() {
    echo -e "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

ok()   { echo -e "${GREEN}OK${NC}"; }
fail() { echo -e "${RED}FAIL${NC}"; }
warn() { echo -e "${YELLOW}WARN${NC}"; }

# --- Helper: run command on a machine (handles jump host for ares-comp-10) ---
run_on() {
    local machine="$1"
    shift
    case "$machine" in
        # Pipe command via stdin to avoid double-SSH quoting issues
        ares-comp-10) echo "$*" | ssh ares "ssh ares-comp-10 bash -s" 2>/dev/null ;;
        *)            ssh "$machine" "$*" 2>/dev/null ;;
    esac
}

# --- Step 1: Deploy latest code ---
log "${CYAN}=== Step 1: Deploying latest code ===${NC}"

deploy_node() {
    local machine="$1"
    log "  Pulling on $machine..."
    if run_on "$machine" "cd ~/$REMOTE_DIR && git pull"; then
        log "  $machine: $(ok)"
    else
        log "  $machine: $(fail) - git pull failed"
        return 1
    fi
}

for machine in homelab chameleon ares; do
    deploy_node "$machine"
done

# ares-comp-10 has no internet — sync from ares via rsync
log "  Syncing to ares-comp-10 via rsync from ares..."
if ssh ares "rsync -a --delete ~/$REMOTE_DIR/ ares-comp-10:~/$REMOTE_DIR/"; then
    log "  ares-comp-10: $(ok)"
else
    log "  ares-comp-10: $(fail) - rsync failed"
fi

# --- Step 2: Clean old verification data ---
log "${CYAN}=== Step 2: Cleaning old data ===${NC}"

for machine in homelab chameleon ares ares-comp-10; do
    run_on "$machine" "mkdir -p ~/drift_data" || true
done
log "  Output directories ensured."

# --- Step 3: Kill any existing collector tmux sessions ---
log "${CYAN}=== Step 3: Stopping any existing collector sessions ===${NC}"

for machine in homelab chameleon ares ares-comp-10; do
    run_on "$machine" "tmux kill-session -t collector 2>/dev/null" || true
done
log "  Old sessions cleaned."

# --- Step 4: Start collectors in tmux ---
log "${CYAN}=== Step 4: Starting collectors (duration=${DURATION}s) ===${NC}"

# Peer clock probing only on ares <-> ares-comp-10 (same LAN).

# homelab: full sensors, no peers
log "  Starting homelab..."
ssh homelab "cd ~/$REMOTE_DIR && \
    tmux new-session -d -s collector \
    'PYTHONPATH=src $PYTHON -m sensor_collector -d $DURATION \
     2>&1 | tee ~/drift_data/collector.log'"
log "  homelab: $(ok)"

# chameleon: sudo, no peers
log "  Starting chameleon..."
ssh chameleon "cd ~/$REMOTE_DIR && \
    tmux new-session -d -s collector \
    'sudo env PYTHONPATH=src $PYTHON -m sensor_collector -d $DURATION \
     -o /home/cc/drift_data \
     2>&1 | tee /home/cc/drift_data/collector.log'"
log "  chameleon: $(ok)"

# ares: no-root, peers with ares-comp-10
log "  Starting ares..."
ssh ares "cd ~/$REMOTE_DIR && \
    tmux new-session -d -s collector \
    'PYTHONPATH=src $PYTHON -m sensor_collector -d $DURATION --no-root-sensors \
     --peers ares-comp-10=172.20.101.10:19777 \
     2>&1 | tee ~/drift_data/collector.log'"
log "  ares: $(ok)"

# ares-comp-10: via jump, no-root, peers with ares
log "  Starting ares-comp-10..."
ssh ares "ssh ares-comp-10 'cd ~/$REMOTE_DIR && \
    tmux new-session -d -s collector \
    \"PYTHONPATH=src $PYTHON -m sensor_collector -d $DURATION --no-root-sensors \
     --peers ares=172.20.1.1:19777 \
     2>&1 | tee ~/drift_data/collector.log\"'"
log "  ares-comp-10: $(ok)"

log ""
log "All collectors started. Monitoring every ${CHECK_INTERVAL}s..."
log ""

# --- Step 5: Monitor loop ---
MACHINES=(homelab chameleon ares ares-comp-10)
declare -A PREV_ROWS

for m in "${MACHINES[@]}"; do
    PREV_ROWS[$m]=0
done

checks_done=0
total_checks=$(( DURATION / CHECK_INTERVAL ))

while (( checks_done < total_checks )); do
    sleep "$CHECK_INTERVAL"
    checks_done=$(( checks_done + 1 ))
    elapsed=$(( checks_done * CHECK_INTERVAL ))

    log "${CYAN}--- Check $checks_done/$total_checks (${elapsed}s elapsed) ---${NC}"

    for machine in "${MACHINES[@]}"; do
        # Check tmux session alive
        if run_on "$machine" "tmux has-session -t collector 2>/dev/null"; then
            status="$(ok)"
        else
            status="$(fail)"
        fi

        # Check CSV row count (newest file only, pipeline avoids nested $())
        row_count=$(run_on "$machine" "ls -t ~/drift_data/*.csv 2>/dev/null | head -1 | xargs cat 2>/dev/null | wc -l" || echo "0")
        row_count="${row_count//[^0-9]/}"  # strip whitespace
        row_count="${row_count:-0}"

        # Calculate rows since last check
        prev="${PREV_ROWS[$machine]:-0}"
        delta=$(( row_count - prev ))
        PREV_ROWS[$machine]=$row_count

        # Check for peer columns (only on first check, newest CSV only)
        if (( checks_done == 1 )); then
            peer_cols=$(run_on "$machine" "ls -t ~/drift_data/*.csv 2>/dev/null | head -1 | xargs head -1 | tr ',' '\n' | grep -c peer" || echo "0")
            peer_cols="${peer_cols//[^0-9]/}"
            peer_cols="${peer_cols:-0}"
            peer_info="peers=${peer_cols}"
        else
            peer_info=""
        fi

        printf "  %-15s session=%-4s  rows=%-6s (+%-4s) %s\n" \
            "$machine" "$status" "$row_count" "$delta" "$peer_info"
    done
    echo ""
done

# --- Step 6: Final summary ---
log "${CYAN}=== Final Summary ===${NC}"
echo ""
printf "%-15s  %8s  %12s  %s\n" "MACHINE" "ROWS" "PEER_COLS" "STATUS"
printf "%-15s  %8s  %12s  %s\n" "-------" "----" "---------" "------"

all_ok=true

for machine in "${MACHINES[@]}"; do
    # Final row count (newest CSV only)
    row_count=$(run_on "$machine" "ls -t ~/drift_data/*.csv 2>/dev/null | head -1 | xargs cat 2>/dev/null | wc -l" || echo "0")
    row_count="${row_count//[^0-9]/}"
    row_count="${row_count:-0}"

    # Peer columns in header (newest CSV only)
    peer_cols=$(run_on "$machine" "ls -t ~/drift_data/*.csv 2>/dev/null | head -1 | xargs head -1 | tr ',' '\n' | grep -c peer" || echo "0")
    peer_cols="${peer_cols//[^0-9]/}"
    peer_cols="${peer_cols:-0}"

    # Determine status (peers only expected on ares/ares-comp-10)
    if (( row_count > 1700 )); then
        status="$(ok)"
    elif (( row_count > 0 )); then
        status="$(warn)"
        all_ok=false
    else
        status="$(fail)"
        all_ok=false
    fi

    printf "%-15s  %8s  %12s  %s\n" "$machine" "$row_count" "$peer_cols" "$status"
done

echo ""

# Print sample peer data from ares pair (only nodes with peers configured)
log "${CYAN}=== Sample Peer Clock Data (last 3 rows) ===${NC}"
echo ""

for machine in ares ares-comp-10; do
    log "  $machine:"
    # Get last 3 values of the peer columns (last N fields via rev|cut|rev)
    sample=$(run_on "$machine" "ls -t ~/drift_data/*.csv 2>/dev/null | head -1 | xargs tail -3 | rev | cut -d, -f1-3 | rev" || echo "")
    if [ -n "$sample" ]; then
        # Print header for context
        hdr=$(run_on "$machine" "ls -t ~/drift_data/*.csv 2>/dev/null | head -1 | xargs head -1 | rev | cut -d, -f1-3 | rev" || echo "")
        log "    $hdr"
        echo "$sample" | while IFS= read -r line; do
            log "    $line"
        done
    else
        log "    No data"
    fi
    echo ""
done

# Show any errors from logs
log "${CYAN}=== Collector Log Errors ===${NC}"
echo ""

for machine in "${MACHINES[@]}"; do
    errors=$(run_on "$machine" "grep -i 'error\|exception\|traceback' ~/drift_data/collector.log 2>/dev/null | tail -5" || true)
    if [ -n "$errors" ]; then
        log "  ${RED}$machine:${NC}"
        echo "$errors" | while IFS= read -r line; do
            log "    $line"
        done
    else
        log "  $machine: no errors"
    fi
done

echo ""
if $all_ok; then
    log "${GREEN}=== VERIFICATION PASSED ===${NC}"
else
    log "${YELLOW}=== VERIFICATION COMPLETED WITH WARNINGS ===${NC}"
fi
