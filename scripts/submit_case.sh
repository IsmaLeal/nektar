#!/usr/bin/env bash
# scripts/submit_case.sh — run a Nektar++ DO case + auto-postprocess, detached.
#
# Usage:
#   scripts/submit_case.sh <case-dir> [--ranks N] [--stages S]
#
# <case-dir> must contain:
#   casefile.xml   the run XML (you write/edit it directly)
#   geometry.xml   the mesh (or a symlink to it)
#
# What it does:
#   1. Validates inputs and snapshots a RUN_SPEC.txt for reproducibility.
#   2. Submits, fully detached:
#        mpirun -np N solver geometry.xml casefile.xml
#        on success → python3 py_utils/workflow.py run-case --case-dir <dir> --stages …
#   3. Returns the driver PID and the paths you'll need.
#
# Auto-detects DOInitModeBasis="Laplacian" and forces --ranks 1 (ARPACK is
# not MPI-aware; see DOVelocityCorrectionScheme.cpp ARPACK init). Override
# with --ranks N. POD and pure VCS runs default to --ranks 3.
#
# After submission you can `exit` the SSH session. When you reconnect:
#   cat <case-dir>/RUN_SPEC.txt          # status (solver_exit, postproc_exit)
#   tail <case-dir>/solver.log           # solver progress
#   tail <case-dir>/postproc.log         # post-processing progress

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOLVER="${SOLVER:-$REPO_ROOT/build/solvers/IncNavierStokesSolver/IncNavierStokesSolver}"
WORKFLOW="${WORKFLOW:-$REPO_ROOT/py_utils/workflow.py}"
DEFAULT_STAGES="extract,derive,suite,check,panels,video"
DEFAULT_NONLAP_RANKS=3

usage() {
    sed -n '2,28p' "$0"
    exit "${1:-0}"
}

# --- parse ---
[[ $# -ge 1 ]] || usage 1
CASE_DIR="$1"; shift
RANKS=""
STAGES="$DEFAULT_STAGES"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ranks)  RANKS="$2";  shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown flag: $1" >&2; usage 1 ;;
    esac
done

# --- validate ---
CASE_DIR="$(realpath "$CASE_DIR")"
[[ -d "$CASE_DIR" ]] || { echo "no such case dir: $CASE_DIR" >&2; exit 1; }
[[ -f "$CASE_DIR/casefile.xml" ]] || { echo "missing $CASE_DIR/casefile.xml" >&2; exit 1; }
[[ -e "$CASE_DIR/geometry.xml" ]] || { echo "missing $CASE_DIR/geometry.xml (file or symlink)" >&2; exit 1; }
[[ -x "$SOLVER" ]] || { echo "no solver: $SOLVER" >&2; exit 1; }
[[ -f "$WORKFLOW" ]] || { echo "no workflow.py: $WORKFLOW" >&2; exit 1; }

# --- ranks: auto-detect ARPACK constraint if not specified ---
if [[ -z "$RANKS" ]]; then
    if grep -qE 'DOInitModeBasis[^"]*"[[:space:]]*VALUE[[:space:]]*=[[:space:]]*"Laplacian"' "$CASE_DIR/casefile.xml" \
       || grep -qE 'PROPERTY[[:space:]]*=[[:space:]]*"DOInitModeBasis"[[:space:]]+VALUE[[:space:]]*=[[:space:]]*"Laplacian"' "$CASE_DIR/casefile.xml"; then
        RANKS=1
        echo "note: detected DOInitModeBasis=Laplacian -> forcing --ranks 1 (ARPACK)"
    else
        RANKS="$DEFAULT_NONLAP_RANKS"
    fi
fi

# --- snapshot the spec (immutable record of what was run) ---
mkdir -p "$CASE_DIR/output"
cat > "$CASE_DIR/RUN_SPEC.txt" <<EOF
$0 $CASE_DIR --ranks $RANKS --stages $STAGES
submitted_at=$(date -Iseconds)
host=$(hostname -s)
solver=$SOLVER
solver_mtime=$(stat -c %y "$SOLVER" 2>/dev/null || echo unknown)
git_HEAD=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo none)
git_dirty=$(git -C "$REPO_ROOT" diff --quiet 2>/dev/null && echo no || echo yes)
casefile_sha=$(sha256sum "$CASE_DIR/casefile.xml" | awk '{print $1}')
ranks=$RANKS
stages=$STAGES
EOF

# --- submit detached ---
nohup bash -c "
    cd '$CASE_DIR'
    mpirun -np $RANKS '$SOLVER' geometry.xml casefile.xml > solver.log 2>&1
    ec=\$?
    echo \"solver_exit=\$ec finished_at=\$(date -Iseconds)\" >> '$CASE_DIR/RUN_SPEC.txt'
    if [ \$ec -eq 0 ]; then
        python3 '$WORKFLOW' run-case --case-dir '$CASE_DIR' --stages '$STAGES' \
            > '$CASE_DIR/postproc.log' 2>&1
        echo \"postproc_exit=\$? postproc_at=\$(date -Iseconds)\" >> '$CASE_DIR/RUN_SPEC.txt'
    else
        echo 'postproc_skipped (solver failed)' >> '$CASE_DIR/RUN_SPEC.txt'
    fi
" > "$CASE_DIR/driver.log" 2>&1 &

PID=$!
disown 2>/dev/null || true
cat <<EOF
submitted: pid=$PID
case dir:  $CASE_DIR
ranks:     $RANKS
stages:    $STAGES
follow:    tail -f $CASE_DIR/solver.log
spec:      $CASE_DIR/RUN_SPEC.txt
EOF
