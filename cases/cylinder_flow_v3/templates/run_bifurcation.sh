#!/usr/bin/env bash
# Cylinder bifurcation sweep: Re=40 (DO+Laplacian), Re=50, Re=100 (DO+POD).
# Re=50/100 first run a deterministic sampling stage to feed POD init.
#
# Usage:
#   cd /scratch2/leal/nektar++
#   bash cases/cylinder_flow_v3/templates/run_bifurcation.sh
#
# Override defaults via env vars, e.g.:
#   NRANKS=3 DO_MODES=4 DO_PARTICLES=20 \
#     bash cases/cylinder_flow_v3/templates/run_bifurcation.sh

set -euo pipefail

# ------------------------------------------------------------------ paths --
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CASE_DIR="$REPO_ROOT/cases/cylinder_flow_v3"
TPL_DIR="$CASE_DIR/templates"
MESH="$CASE_DIR/mesh/geometry.xml"
SOLVER="${SOLVER:-$REPO_ROOT/build/dist/bin/IncNavierStokesSolver}"

if [[ ! -x "$SOLVER" ]]; then
    echo "ERROR: solver not found at $SOLVER" >&2
    echo "       set SOLVER=/path/to/IncNavierStokesSolver and re-run." >&2
    exit 1
fi
if [[ ! -f "$MESH" ]]; then
    echo "ERROR: mesh not found at $MESH" >&2
    exit 1
fi

# ------------------------------------------------------------ parameters --
DT="${DT:-0.01}"
T_END="${T_END:-50.0}"
CHK_STEPS="${CHK_STEPS:-50}"      # 50 * 0.01 = 0.5 s between snapshots
INFO_STEPS="${INFO_STEPS:-250}"    # 2.5 s between info dumps
DO_MODES="${DO_MODES:-4}"
DO_PARTICLES="${DO_PARTICLES:-20}"
NRANKS="${NRANKS:-3}"

STAMP="$(date +%Y%m%d-%H%M%S)"
SWEEP_DIR="$CASE_DIR/runs/bifurcation_${STAMP}"
mkdir -p "$SWEEP_DIR"
echo "[run_bifurcation] sweep root: $SWEEP_DIR"

# ------------------------------------------------------------- helpers ---
substitute() {
    # $1 = template path, $2 = output path, $3.. = KEY=VAL pairs
    local in="$1" out="$2"; shift 2
    local sed_args=()
    for kv in "$@"; do
        local k="${kv%%=*}" v="${kv#*=}"
        sed_args+=( -e "s|@${k}@|${v}|g" )
    done
    sed "${sed_args[@]}" "$in" > "$out"
}

run_sampling() {
    local re="$1" workdir="$2"
    mkdir -p "$workdir/output"
    substitute "$TPL_DIR/sampling.xml.in" "$workdir/casefile.xml" \
        "RE=$re" "DT=$DT" "T_END=$T_END" \
        "CHK_STEPS=$CHK_STEPS" "INFO_STEPS=$INFO_STEPS"
    ln -sf "$MESH" "$workdir/geometry.xml"
    echo "[run_bifurcation] sampling Re=$re  (mpirun -np $NRANKS)"
    ( cd "$workdir" && \
      mpirun -np "$NRANKS" "$SOLVER" casefile.xml geometry.xml \
        > sampling.log 2>&1 )
    echo "[run_bifurcation] sampling Re=$re  done"
}

run_do() {
    # mode = "pod" or "lap"; pod_pattern only used if mode=pod
    local re="$1" workdir="$2" mode="$3" pod_pattern="${4:-}"
    mkdir -p "$workdir/output"
    if [[ "$mode" == "pod" ]]; then
        substitute "$TPL_DIR/do_pod.xml.in" "$workdir/casefile.xml" \
            "RE=$re" "DT=$DT" "T_END=$T_END" \
            "CHK_STEPS=$CHK_STEPS" "INFO_STEPS=$INFO_STEPS" \
            "DO_MODES=$DO_MODES" "DO_PARTICLES=$DO_PARTICLES" \
            "POD_PATTERN=$pod_pattern"
    else
        substitute "$TPL_DIR/do_lap.xml.in" "$workdir/casefile.xml" \
            "RE=$re" "DT=$DT" "T_END=$T_END" \
            "CHK_STEPS=$CHK_STEPS" "INFO_STEPS=$INFO_STEPS" \
            "DO_MODES=$DO_MODES" "DO_PARTICLES=$DO_PARTICLES"
    fi
    ln -sf "$MESH" "$workdir/geometry.xml"
    echo "[run_bifurcation] DO Re=$re  init=$mode  (mpirun -np $NRANKS)"
    ( cd "$workdir" && \
      mpirun -np "$NRANKS" "$SOLVER" casefile.xml geometry.xml \
        > do.log 2>&1 )
    echo "[run_bifurcation] DO Re=$re  init=$mode  done"
}

# --------------------------------------------------------------- runs ----

# Re = 40 : DO with Laplacian init, no sampling needed.
RE40="$SWEEP_DIR/Re040_lap"
mkdir -p "$RE40"
run_do 40 "$RE40/do" "lap"

# Re = 50 : sampling -> DO with POD init.
RE50="$SWEEP_DIR/Re050_pod"
mkdir -p "$RE50"
run_sampling 50 "$RE50/sampling"
run_do 50 "$RE50/do" "pod" "../sampling/output/casefile_*.chk"

# Re = 100 : sampling -> DO with POD init.
RE100="$SWEEP_DIR/Re100_pod"
mkdir -p "$RE100"
run_sampling 100 "$RE100/sampling"
run_do 100 "$RE100/do" "pod" "../sampling/output/casefile_*.chk"

# ---------------------------------------------------------- summary ----
echo
echo "==================== sweep complete ===================="
echo "outputs:"
for d in "$RE40/do" "$RE50/do" "$RE100/do"; do
    n=$(ls "$d/output/"casefile.do_*.fld 2>/dev/null | wc -l)
    echo "  $d  ->  $n  .fld snapshots"
done
echo "sweep root: $SWEEP_DIR"
