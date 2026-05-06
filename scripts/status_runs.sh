#!/usr/bin/env bash
# scripts/status_runs.sh — show submission status for every run dir under a root.
# Usage: scripts/status_runs.sh [run-root]    # default: cases/*/runs/*
#
# Reads RUN_SPEC.txt in each candidate dir and prints one summary line per run.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-$REPO_ROOT/cases/*/runs/*}"

shopt -s nullglob
printf "%-50s  %-8s  %-12s  %-12s  %s\n" "case-dir" "ranks" "solver_exit" "postproc_exit" "submitted_at"
printf "%-50s  %-8s  %-12s  %-12s  %s\n" "$(printf '%.0s-' {1..50})" "$(printf '%.0s-' {1..8})" "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..12})" "-------------------"
for d in $ROOT $ROOT/do; do
    [[ -f "$d/RUN_SPEC.txt" ]] || continue
    spec="$d/RUN_SPEC.txt"
    sub=$(sed -n 's/^submitted_at=//p' "$spec" | head -1)
    rks=$(sed -n 's/^ranks=//p' "$spec" | head -1)
    sex=$(sed -n 's/^solver_exit=\([0-9]*\).*/\1/p' "$spec" | head -1)
    pex=$(sed -n 's/^postproc_exit=\([0-9]*\).*/\1/p' "$spec" | head -1)
    [[ -z "$sex" ]] && sex="(running)"
    [[ -z "$pex" && -n "$sex" && "$sex" != "(running)" ]] && pex="(skipped)"
    [[ -z "$pex" ]] && pex="(pending)"
    short="${d#$REPO_ROOT/}"
    printf "%-50s  %-8s  %-12s  %-12s  %s\n" "${short:0:50}" "$rks" "$sex" "$pex" "$sub"
done
