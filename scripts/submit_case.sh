#!/usr/bin/env bash
# scripts/submit_case.sh — run a Nektar++ case + auto-postprocess, detached.
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
#   2. Auto-detects:
#        - DOInitModeBasis="Laplacian"            -> --ranks 1 (ARPACK constraint)
#        - SolverType="VelocityCorrectionScheme"  -> sampling (deterministic NS)
#        - SolverType="DOVelocityCorrectionScheme"-> DO run
#   3. Submits, fully detached:
#        mpirun -np N solver geometry.xml casefile.xml
#        on success →
#          DO       run: python3 py_utils/workflow.py run-case --stages …
#          sampling run: python3 py_utils/load_chk.py + animate_single.py
#                        (extract on actual domain bounds, hi-res grid, single
#                         realisation video — same look as do_panels' real panel)
#
# Sampling-postproc env-var overrides (defaults work for cylinder_v3):
#   SAMP_NX           grid points in x (default 900)
#   SAMP_NY           grid points in y (default 600)
#   SAMP_BOUNDS       "xmin,xmax,ymin,ymax"  (default: auto from geometry.xml)
#   SAMP_DURATION_SEC video playback length in seconds (default 13)
#   SAMP_DPI          render dpi (default 200)
#   SAMP_FIGSIZE      figsize WxH inches (default 13x6.5)
#
# After submission you can `exit` the SSH session. When you reconnect:
#   cat <case-dir>/RUN_SPEC.txt          # status
#   tail <case-dir>/solver.log           # solver progress
#   tail <case-dir>/postproc.log         # post-processing progress

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOLVER="${SOLVER:-$REPO_ROOT/build/solvers/IncNavierStokesSolver/IncNavierStokesSolver}"
WORKFLOW="${WORKFLOW:-$REPO_ROOT/py_utils/workflow.py}"
LOAD_CHK="${LOAD_CHK:-$REPO_ROOT/py_utils/load_chk.py}"
ANIMATE_SINGLE="${ANIMATE_SINGLE:-$REPO_ROOT/py_utils/reconstruction/animation/animate_single.py}"
DEFAULT_STAGES="extract,derive,suite,check,panels,video"
DEFAULT_NONLAP_RANKS=3

# sampling postproc defaults (overridable via env vars)
SAMP_NX="${SAMP_NX:-900}"
SAMP_NY="${SAMP_NY:-600}"
SAMP_DURATION_SEC="${SAMP_DURATION_SEC:-13}"
SAMP_DPI="${SAMP_DPI:-200}"
SAMP_FIGSIZE="${SAMP_FIGSIZE:-13x6.5}"
SAMP_BOUNDS="${SAMP_BOUNDS:-}"   # empty = auto-detect from geometry.xml

usage() {
    sed -n '2,38p' "$0"
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

# --- detect sampling vs DO from SolverType ---
IS_SAMPLING=0
if grep -qE 'SolverType[^"]*"[[:space:]]*VALUE[[:space:]]*=[[:space:]]*"DOVelocityCorrectionScheme"' "$CASE_DIR/casefile.xml" \
   || grep -qE 'PROPERTY[[:space:]]*=[[:space:]]*"SolverType"[[:space:]]+VALUE[[:space:]]*=[[:space:]]*"DOVelocityCorrectionScheme"' "$CASE_DIR/casefile.xml"; then
    IS_SAMPLING=0
elif grep -qE 'SolverType[^"]*"[[:space:]]*VALUE[[:space:]]*=[[:space:]]*"VelocityCorrectionScheme"' "$CASE_DIR/casefile.xml" \
   || grep -qE 'PROPERTY[[:space:]]*=[[:space:]]*"SolverType"[[:space:]]+VALUE[[:space:]]*=[[:space:]]*"VelocityCorrectionScheme"' "$CASE_DIR/casefile.xml"; then
    IS_SAMPLING=1
fi

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

# --- sampling: resolve domain bounds (auto from geometry.xml unless overridden) ---
if [[ "$IS_SAMPLING" -eq 1 && -z "$SAMP_BOUNDS" ]]; then
    SAMP_BOUNDS="$(python3 - "$CASE_DIR/geometry.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
xs, ys = [], []
for v in root.iter('V'):
    if v.text is None: continue
    parts = v.text.split()
    if len(parts) >= 2:
        try: xs.append(float(parts[0])); ys.append(float(parts[1]))
        except ValueError: pass
if not xs or not ys:
    print(""); sys.exit(0)
print(f"{min(xs)},{max(xs)},{min(ys)},{max(ys)}")
PY
)"
    if [[ -z "$SAMP_BOUNDS" ]]; then
        echo "warning: could not auto-detect domain bounds from geometry.xml; set SAMP_BOUNDS=xmin,xmax,ymin,ymax to override" >&2
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
is_sampling=$IS_SAMPLING
EOF
if [[ "$IS_SAMPLING" -eq 1 ]]; then
    cat >> "$CASE_DIR/RUN_SPEC.txt" <<EOF
samp_bounds=$SAMP_BOUNDS
samp_grid=${SAMP_NX}x${SAMP_NY}
samp_video_dpi=$SAMP_DPI
samp_video_figsize=$SAMP_FIGSIZE
samp_video_duration_sec=$SAMP_DURATION_SEC
EOF
else
    cat >> "$CASE_DIR/RUN_SPEC.txt" <<EOF
stages=$STAGES
EOF
fi

# --- build the post-processing command (sampling vs DO branches) ---
if [[ "$IS_SAMPLING" -eq 1 ]]; then
    # Parse SAMP_BOUNDS into individual xmin xmax ymin ymax for load_chk
    IFS=',' read -r SXMIN SXMAX SYMIN SYMAX <<< "$SAMP_BOUNDS"
    POSTPROC_CMD=$(cat <<EOF
python3 '$LOAD_CHK' \
    --out '$CASE_DIR/output/out.h5' \
    --xml '$CASE_DIR/casefile.xml' \
    --chk-dir '$CASE_DIR/output' \
    --xmin=$SXMIN --xmax $SXMAX --ymin=$SYMIN --ymax $SYMAX \
    --nx $SAMP_NX --ny $SAMP_NY \
  && python3 '$ANIMATE_SINGLE' \
    --data '$CASE_DIR/output/out.h5' \
    --out '$CASE_DIR/output/py_utils_results/vorticity.mp4' \
    --field vorticity \
    --duration-sec $SAMP_DURATION_SEC \
    --dpi $SAMP_DPI --figsize $SAMP_FIGSIZE
EOF
)
else
    POSTPROC_CMD="python3 '$WORKFLOW' run-case --case-dir '$CASE_DIR' --stages '$STAGES'"
fi

# --- submit detached ---
nohup bash -c "
    cd '$CASE_DIR'
    mpirun -np $RANKS '$SOLVER' geometry.xml casefile.xml > solver.log 2>&1
    ec=\$?
    echo \"solver_exit=\$ec finished_at=\$(date -Iseconds)\" >> '$CASE_DIR/RUN_SPEC.txt'
    if [ \$ec -eq 0 ]; then
        $POSTPROC_CMD > '$CASE_DIR/postproc.log' 2>&1
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
mode:      $([[ "$IS_SAMPLING" -eq 1 ]] && echo "sampling (auto extract+single-video)" || echo "DO (workflow.py stages=$STAGES)")
follow:    tail -f $CASE_DIR/solver.log
spec:      $CASE_DIR/RUN_SPEC.txt
EOF
