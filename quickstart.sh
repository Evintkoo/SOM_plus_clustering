#!/bin/bash
# Quick start script for SOM++ Clustering Dashboard

set -e

TIMESTAMP() { date '+%Y-%m-%dT%H:%M:%S'; }

log_info()  { echo "[$(TIMESTAMP)] [INFO]  $1"; }
log_ok()    { echo "[$(TIMESTAMP)] [OK]    $1"; }
log_warn()  { echo "[$(TIMESTAMP)] [WARN]  $1"; }
log_error() { echo "[$(TIMESTAMP)] [ERROR] $1"; }

# Cleanup on exit
cleanup() {
    echo ""
    log_warn "Shutting down services..."
    [ -n "$API_PID" ] && kill $API_PID 2>/dev/null && log_info "API stopped (PID: $API_PID)"
    [ -n "$UI_PID" ] && kill $UI_PID 2>/dev/null && log_info "UI stopped (PID: $UI_PID)"
    log_ok "All services stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

log_info "SOM++ Quickstart initializing..."
echo ""

log_info "[1/3] Building Rust backend..."
cargo build --release 2>&1 | grep -E "Compiling|Finished|error" || true
log_ok "Backend built (release mode)"
echo ""

log_info "[2/3] Starting REST API on port 3000..."
cd som-api
cargo run --release &
API_PID=$!
cd ..
sleep 2
log_ok "API running (PID: $API_PID, addr: 127.0.0.1:3000)"
echo ""

log_info "[3/3] Starting React UI on port 5173..."
cd som-ui
npm install --silent 2>/dev/null
npm run dev &
UI_PID=$!
cd ..
sleep 3
log_ok "UI running (PID: $UI_PID, addr: http://localhost:5173)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_ok "SOM++ Dashboard ready"
echo ""
echo "  Dashboard:   http://localhost:5173"
echo "  API:         http://localhost:3000"
echo "  Health:      curl http://localhost:3000/health"
echo ""
echo "  Ctrl+C to stop all services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep script running
wait
