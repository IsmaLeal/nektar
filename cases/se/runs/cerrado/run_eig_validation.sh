#!/usr/bin/env bash
# Eigenvalue-vs-Re validation campaign, 2026-07-17.
# Phase A: symmetric bases at Re = 70, 90, 110 (in parallel).
# Phase B: kick DNS (70, 90, 110) + unforced DO (70, 79.3, 90, 110),
#          max MAXPAR concurrent serial jobs, longest first.
# Everything runs serial (Laplacian DO init is single-rank) and nice'd so
# the production DO continuation on 4 cores is not disturbed.
set -u

BIN=/home/isma/nektar_repro/build_fast/solvers/IncNavierStokesSolver/IncNavierStokesSolver
RUNS=/home/isma/nektar_src_full/cases/se/runs/cerrado
LOG=$RUNS/2026_07_17_eig_campaign.log
MAXPAR=3

BASES=(2026_07_17_eig_base_Re70 2026_07_17_eig_base_Re90 2026_07_17_eig_base_Re110)
# longest first so the pool drains well
PHASE2=(2026_07_17_eig_do_Re79p3 2026_07_17_eig_do_Re90 2026_07_17_eig_do_Re70
        2026_07_17_eig_dns_Re90 2026_07_17_eig_dns_Re70 2026_07_17_eig_dns_Re110
        2026_07_17_eig_do_Re110)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_case() {
    local d=$1
    log "START $d"
    ( cd "$RUNS/$d" && nice -n 10 "$BIN" geometry.xml casefile.xml > run.log 2>&1 )
    local rc=$?
    log "DONE  $d (exit $rc)"
    return $rc
}

last_chk() { ls -v "$RUNS/$1"/output/chks/casefile_*.chk 2>/dev/null | tail -1; }

log "=== Phase A: base flows ==="
for d in "${BASES[@]}"; do run_case "$d" & done
wait

# seed phase-B ICs from the base final checkpoints
for re in 70 90 110; do
    c=$(last_chk "2026_07_17_eig_base_Re$re")
    if [ -z "$c" ]; then
        log "ERROR base Re$re produced no chk; its DNS/DO runs will be skipped"
        continue
    fi
    cp "$c" "$RUNS/2026_07_17_eig_dns_Re$re/ic.chk"
    [ -d "$RUNS/2026_07_17_eig_do_Re$re" ] && cp "$c" "$RUNS/2026_07_17_eig_do_Re$re/ic.chk"
    log "IC Re$re <- $(basename "$c")"
done

log "=== Phase B: kick DNS + unforced DO ==="
running=0
for d in "${PHASE2[@]}"; do
    if [ ! -f "$RUNS/$d/ic.chk" ] && [ ! -f "$RUNS/$d/ic.fld" ]; then
        log "SKIP  $d (no IC)"
        continue
    fi
    run_case "$d" &
    running=$((running + 1))
    if [ "$running" -ge "$MAXPAR" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

log "=== Campaign finished ==="
